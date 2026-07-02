/**
 * RfConstellation — live 3D constellation of the whole RF/network environment:
 * the connected AP, every nearby AP (positioned by real RSSI), and LAN devices
 * discovered on the network. Driven by the additive `rf_world` field on the
 * sensing stream. Real data only — no placeholders.
 */
import * as THREE from 'three';

const KIND_COLOR = {
  ap_connected: 0x2090ff, // connected AP — matches the router signal blue
  ap_nearby:    0x00d878, // neighbor APs — green
  lan_device:   0xffb020, // LAN devices — amber
};

// stronger signal = closer to the observer/router origin
function rssiToRadius(rssi) {
  const c = Math.max(-90, Math.min(-40, rssi));
  return 1.8 + ((c - -40) / -50) * 6.0; // 1.8 .. 7.8
}

function bssidHash(id) {
  let h = 0;
  for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) & 0xffff;
  return h / 0xffff;
}

export class RfConstellation {
  constructor(scene) {
    this._scene = scene;
    this._nodes = new Map(); // id -> entry
    this._group = new THREE.Group();
    scene.add(this._group);
  }

  update(data, elapsed) {
    const world = data?.rf_world;
    if (!world || world.length === 0) { this._fadeAll(); return; }

    const aps = world.filter((n) => n.kind !== 'lan_device');
    const lan = world.filter((n) => n.kind === 'lan_device');
    const N = Math.max(aps.length, 1);
    aps.forEach((node, i) => {
      const az = (i / N) * Math.PI * 2 + bssidHash(node.id) * 0.3;
      const r = node.rssi != null ? rssiToRadius(node.rssi) : 5.0;
      this._upsert(node, Math.cos(az) * r, 1.2 + (node.kind === 'ap_connected' ? 0.6 : 0), Math.sin(az) * r);
    });
    const M = Math.max(lan.length, 1);
    lan.forEach((node, i) => {
      const az = (i / M) * Math.PI * 2 + 0.4;
      this._upsert(node, Math.cos(az) * 2.6, 0.18, Math.sin(az) * 2.6);
    });

    this._tick(elapsed);
    this._cull(world.map((n) => n.id));
  }

  _upsert(node, tx, ty, tz) {
    const color = KIND_COLOR[node.kind] ?? 0xffffff;
    if (!this._nodes.has(node.id)) {
      const isConn = node.kind === 'ap_connected';
      const mesh = new THREE.Mesh(
        new THREE.SphereGeometry(isConn ? 0.17 : 0.11, 16, 12),
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9 })
      );
      // additive halo
      const rMat = new THREE.MeshBasicMaterial({
        color, transparent: true, opacity: 0.28, side: THREE.DoubleSide,
        wireframe: true, blending: THREE.AdditiveBlending, depthWrite: false,
      });
      const ring = new THREE.Mesh(new THREE.SphereGeometry(0.26, 18, 12), rMat);
      mesh.add(ring);
      const linGeo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0.9, 0), new THREE.Vector3(tx, ty, tz),
      ]);
      const linMat = new THREE.LineBasicMaterial({
        color, transparent: true, opacity: 0.22, blending: THREE.AdditiveBlending,
      });
      const line = new THREE.Line(linGeo, linMat);
      this._group.add(mesh);
      this._group.add(line);
      this._nodes.set(node.id, {
        mesh, ring, line, mat: mesh.material, rMat, linMat, linGeo,
        cur: new THREE.Vector3(tx, ty, tz), tgt: new THREE.Vector3(tx, ty, tz),
        lastSeen: performance.now(), rssi: node.rssi,
      });
    }
    const e = this._nodes.get(node.id);
    e.tgt.set(tx, ty, tz);
    e.rssi = node.rssi;
    e.lastSeen = performance.now();
  }

  _tick(elapsed) {
    const k = 0.06;
    for (const [, e] of this._nodes) {
      e.cur.lerp(e.tgt, k);
      e.mesh.position.copy(e.cur);
      e.ring.rotation.y += 0.02;
      // gentle breathing pulse on the halo, brighter for stronger signal
      const s = 0.85 + 0.15 * Math.sin(elapsed * 2.2 + e.cur.x);
      e.ring.scale.setScalar(s);
      const pts = e.linGeo.attributes.position;
      pts.setXYZ(1, e.cur.x, e.cur.y, e.cur.z);
      pts.needsUpdate = true;
    }
  }

  _cull(activeIds) {
    const now = performance.now();
    for (const [id, e] of this._nodes) {
      if (!activeIds.includes(id) && now - e.lastSeen > 5000) {
        this._group.remove(e.mesh);
        this._group.remove(e.line);
        e.mesh.geometry.dispose(); e.mat.dispose(); e.rMat.dispose();
        e.linGeo.dispose(); e.linMat.dispose();
        this._nodes.delete(id);
      }
    }
  }

  _fadeAll() {
    for (const [, e] of this._nodes) e.mat.opacity *= 0.95;
  }
}
