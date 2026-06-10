//! Responder-side setup registry for the 802.11bf sensing model — enforces
//! the setup-ID-collision and capacity rejection paths a single session
//! cannot see on its own (ADR-153 acceptance: duplicate setup ID rejected).

use std::collections::BTreeMap;

use super::messages::{SensingMeasurementSetupRequest, SensingMeasurementSetupResponse};
use super::session::{Action, SensingSession, SessionConfig, SessionEvent, SessionState};
use super::types::{BfError, MeasurementSetupId, SetupStatus};

/// Responder-side registry of sensing sessions keyed by setup ID.
///
/// Enforces the setup-ID-collision and capacity rejection paths the single
/// session cannot see on its own.
#[derive(Debug)]
pub struct SessionTable {
    config: SessionConfig,
    sessions: BTreeMap<u8, SensingSession>,
}

impl SessionTable {
    pub fn new(config: SessionConfig) -> Self {
        Self {
            config,
            sessions: BTreeMap::new(),
        }
    }

    /// Number of setups not in Idle.
    pub fn active_setups(&self) -> usize {
        self.sessions
            .values()
            .filter(|s| s.state() != SessionState::Idle)
            .count()
    }

    pub fn session(&self, setup_id: MeasurementSetupId) -> Option<&SensingSession> {
        self.sessions.get(&setup_id.value())
    }

    /// Route an inbound setup request, rejecting setup-ID collisions and
    /// capacity overruns before delegating to a responder session.
    pub fn handle_setup_request(
        &mut self,
        req: SensingMeasurementSetupRequest,
    ) -> Result<Vec<Action>, BfError> {
        let reject = |setup_id, status| {
            Ok(vec![Action::SendSetupResponse(
                SensingMeasurementSetupResponse { setup_id, status },
            )])
        };
        if let Some(existing) = self.sessions.get(&req.setup_id.value()) {
            if existing.state() != SessionState::Idle {
                return reject(req.setup_id, SetupStatus::RejectedSetupIdCollision);
            }
        }
        if self.active_setups() >= self.config.capabilities.max_active_setups as usize {
            return reject(req.setup_id, SetupStatus::RejectedCapacity);
        }
        let key = req.setup_id.value();
        let mut session = SensingSession::new_responder(self.config.clone());
        let actions = session.handle(SessionEvent::SetupRequestReceived(req))?;
        self.sessions.insert(key, session);
        Ok(actions)
    }

    /// Route any other event to the session owning `setup_id` (no-op if the
    /// setup is unknown — stray frames are ignored, not errors).
    pub fn handle_for(
        &mut self,
        setup_id: MeasurementSetupId,
        event: SessionEvent,
    ) -> Result<Vec<Action>, BfError> {
        match self.sessions.get_mut(&setup_id.value()) {
            Some(session) => session.handle(event),
            None => Ok(vec![]),
        }
    }
}
