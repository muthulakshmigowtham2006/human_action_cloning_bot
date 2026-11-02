import React, { useEffect, useRef, useState } from 'react'

/**
 * 4x4 Homogeneous Matrix Helpers (Row-major)
 */
function matIdentity() {
  return [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ]
}

function matMul(A, B) {
  const C = matIdentity()
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      C[i][j] =
        A[i][0] * B[0][j] +
        A[i][1] * B[1][j] +
        A[i][2] * B[2][j] +
        A[i][3] * B[3][j]
    }
  }
  return C
}

function matTranslate(x, y, z) {
  return [
    [1, 0, 0, x],
    [0, 1, 0, y],
    [0, 0, 1, z],
    [0, 0, 0, 1],
  ]
}

function matRotZ(theta) {
  const c = Math.cos(theta)
  const s = Math.sin(theta)
  return [
    [c, -s, 0, 0],
    [s, c, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ]
}

function matRotY(theta) {
  const c = Math.cos(theta)
  const s = Math.sin(theta)
  return [
    [c, 0, s, 0],
    [0, 1, 0, 0],
    [-s, 0, c, 0],
    [0, 0, 0, 1],
  ]
}

function transformPoint(M, p) {
  const [x, y, z] = p
  return [
    M[0][0] * x + M[0][1] * y + M[0][2] * z + M[0][3],
    M[1][0] * x + M[1][1] * y + M[1][2] * z + M[1][3],
    M[2][0] * x + M[2][1] * y + M[2][2] * z + M[2][3],
  ]
}

function subVec(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

function crossVec(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]
}

function normVec(v) {
  const len = Math.hypot(v[0], v[1], v[2]) || 1
  return [v[0] / len, v[1] / len, v[2] / len]
}

function dotVec(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

function createBoxFaces(T, size, originXyz, rgba) {
  const [sx, sy, sz] = size
  const [ox, oy, oz] = originXyz
  const hx = sx / 2
  const hy = sy / 2
  const hz = sz / 2

  const localVerts = [
    [ox - hx, oy - hy, oz - hz],
    [ox + hx, oy - hy, oz - hz],
    [ox + hx, oy + hy, oz - hz],
    [ox - hx, oy + hy, oz - hz],
    [ox - hx, oy - hy, oz + hz],
    [ox + hx, oy - hy, oz + hz],
    [ox + hx, oy + hy, oz + hz],
    [ox - hx, oy + hy, oz + hz],
  ]

  const worldVerts = localVerts.map((v) => transformPoint(T, v))
  const indices = [
    [0, 3, 2, 1], // bottom
    [4, 5, 6, 7], // top
    [0, 1, 5, 4], // front -Y
    [2, 3, 7, 6], // back +Y
    [1, 2, 6, 5], // +X
    [3, 0, 4, 7], // -X
  ]

  return indices.map((quad) => ({
    verts: quad.map((i) => worldVerts[i]),
    rgba,
  }))
}

function createCylinderFaces(T, radius, length, originXyz, rgba, segments = 18) {
  const [ox, oy, oz] = originXyz
  const z0 = oz - length / 2
  const z1 = oz + length / 2
  const faces = []

  const botRing = []
  const topRing = []
  for (let i = 0; i < segments; i++) {
    const ang = (i / segments) * Math.PI * 2
    const cx = ox + radius * Math.cos(ang)
    const cy = oy + radius * Math.sin(ang)
    botRing.push(transformPoint(T, [cx, cy, z0]))
    topRing.push(transformPoint(T, [cx, cy, z1]))
  }

  for (let i = 0; i < segments; i++) {
    const next = (i + 1) % segments
    faces.push({
      verts: [botRing[i], botRing[next], topRing[next], topRing[i]],
      rgba,
    })
  }
  faces.push({ verts: topRing, rgba })
  return faces
}

export default function RobotViewport3D({
  urdfSpec = null,
  liveJointsRad = null,
  targetJointsRad = null,
  isGazeboRunning = false,
  isControllerReady = false,
  executionStatus = 'ready',
  waypoints = [],
  executedTrail = [],
  currentStep = 0,
  wsConnected = false,
}) {
  const canvasRef = useRef(null)

  // Interactive Orbit Camera State
  const [camOrbit, setCamOrbit] = useState({
    azimuth: 0.68, // rad
    elevation: 0.36, // rad
    distance: 4.15, // m
    targetZ: 1.32, // m
  })
  const dragRef = useRef({ active: false, lastX: 0, lastY: 0 })

  // Optional 3D Trajectory Preview Mode (when not actively executing in Gazebo)
  const [previewPlaying, setPreviewPlaying] = useState(false)
  const [previewIdx, setPreviewIdx] = useState(0)

  useEffect(() => {
    if (!previewPlaying || waypoints.length === 0) return
    const timer = setInterval(() => {
      setPreviewIdx((prev) => {
        if (prev + 1 >= waypoints.length) {
          setPreviewPlaying(false)
          return 0
        }
        return prev + 1
      })
    }, 320)
    return () => clearInterval(timer)
  }, [previewPlaying, waypoints])

  // Stop local preview automatically if real Gazebo execution starts
  useEffect(() => {
    if (executionStatus === 'executing') {
      setPreviewPlaying(false)
    }
  }, [executionStatus])

  // Determine which joint angles are currently driving the 3D URDF arm:
  // 1. If previewPlaying is active, show the preview waypoint joints (clearly badged as PREVIEW).
  // 2. Otherwise, if Gazebo is running & publishing /joint_states, show actual liveJointsRad.
  // 3. Otherwise, show [0, 0, 0] home pose.
  const activeJoints =
    previewPlaying && waypoints[previewIdx]?.joints_rad
      ? waypoints[previewIdx].joints_rad
      : liveJointsRad && liveJointsRad.length === 3
      ? liveJointsRad
      : [0.0, 0.0, 0.0]

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const W = canvas.width
    const H = canvas.height

    // 1. Background gradient matching dark robotics theme
    const grad = ctx.createLinearGradient(0, 0, 0, H)
    grad.addColorStop(0, '#070d19')
    grad.addColorStop(1, '#0f172a')
    ctx.fillStyle = grad
    ctx.fillRect(0, 0, W, H)

    // 2. Camera Matrix (Orbit around [0, 0, camOrbit.targetZ])
    const { azimuth, elevation, distance, targetZ } = camOrbit
    const camX = distance * Math.cos(elevation) * Math.cos(azimuth)
    const camY = distance * Math.cos(elevation) * Math.sin(azimuth)
    const camZ = targetZ + distance * Math.sin(elevation)
    const camPos = [camX, camY, camZ]
    const lookAt = [0, 0, targetZ]

    const forward = normVec(subVec(lookAt, camPos))
    const worldUp = [0, 0, 1]
    const right = normVec(crossVec(forward, worldUp))
    const up = crossVec(right, forward)
    const focal = Math.min(W, H) * 1.05

    const project3D = (pt) => {
      const rel = subVec(pt, camPos)
      const xCam = dotVec(rel, right)
      const yCam = dotVec(rel, up)
      const zCam = dotVec(rel, forward)
      if (zCam <= 0.15) return null
      const sx = W / 2 + (xCam / zCam) * focal
      const sy = H / 2 - (yCam / zCam) * focal
      return { x: sx, y: sy, depth: zCam }
    }

    // 3. Draw Ground Grid (Z = 0 plane from robot_world.sdf)
    ctx.lineWidth = 1
    for (let g = -2.0; g <= 2.001; g += 0.4) {
      const isAxis = Math.abs(g) < 0.001
      ctx.strokeStyle = isAxis ? 'rgba(148, 163, 184, 0.28)' : 'rgba(51, 65, 85, 0.42)'
      const p1 = project3D([g, -2.0, 0])
      const p2 = project3D([g, 2.0, 0])
      if (p1 && p2) {
        ctx.beginPath()
        ctx.moveTo(p1.x, p1.y)
        ctx.lineTo(p2.x, p2.y)
        ctx.stroke()
      }
      const q1 = project3D([-2.0, g, 0])
      const q2 = project3D([2.0, g, 0])
      if (q1 && q2) {
        ctx.beginPath()
        ctx.moveTo(q1.x, q1.y)
        ctx.lineTo(q2.x, q2.y)
        ctx.stroke()
      }
    }

    // Draw World Coordinate Frame Axes at origin (X=Red, Y=Green, Z=Blue)
    const origin2D = project3D([0, 0, 0])
    const xAxis2D = project3D([0.55, 0, 0])
    const yAxis2D = project3D([0, 0.55, 0])
    const zAxis2D = project3D([0, 0, 0.55])
    if (origin2D && xAxis2D && yAxis2D && zAxis2D) {
      ctx.lineWidth = 2
      ctx.strokeStyle = '#ef4444'
      ctx.beginPath()
      ctx.moveTo(origin2D.x, origin2D.y)
      ctx.lineTo(xAxis2D.x, xAxis2D.y)
      ctx.stroke()
      ctx.fillStyle = '#ef4444'
      ctx.font = 'bold 10px JetBrains Mono, monospace'
      ctx.fillText('+X', xAxis2D.x + 3, xAxis2D.y)

      ctx.strokeStyle = '#10b981'
      ctx.beginPath()
      ctx.moveTo(origin2D.x, origin2D.y)
      ctx.lineTo(yAxis2D.x, yAxis2D.y)
      ctx.stroke()
      ctx.fillStyle = '#10b981'
      ctx.fillText('+Y', yAxis2D.x + 3, yAxis2D.y)

      ctx.strokeStyle = '#38bdf8'
      ctx.beginPath()
      ctx.moveTo(origin2D.x, origin2D.y)
      ctx.lineTo(zAxis2D.x, zAxis2D.y)
      ctx.stroke()
      ctx.fillStyle = '#38bdf8'
      ctx.fillText('+Z', zAxis2D.x + 3, zAxis2D.y)
    }

    // 4. Compute exact URDF Link Transforms from activeJoints [q1, q2, q3]
    const [q1, q2, q3] = activeJoints
    const links = urdfSpec?.links || {}

    // base_link at world origin
    const T_base = matIdentity()
    // joint1: origin [0, 0, 0.2], axis [0, 0, 1] -> link1
    const T_j1 = matMul(T_base, matMul(matTranslate(0, 0, 0.2), matRotZ(q1)))
    // joint2: origin [0, 0, 1.0], axis [0, 1, 0] -> link2
    const T_j2 = matMul(T_j1, matMul(matTranslate(0, 0, 1.0), matRotY(q2)))
    // joint3: origin [0, 0, 0.9], axis [0, 1, 0] -> link3
    const T_j3 = matMul(T_j2, matMul(matTranslate(0, 0, 0.9), matRotY(q3)))
    // end_effector_joint: fixed origin [0, 0, 0.8] -> end_effector
    const T_ee = matMul(T_j3, matTranslate(0, 0, 0.8))

    const allFaces = []

    // Base link cylinder (radius 0.3, length 0.2, origin [0, 0, 0.1])
    const baseVis = links.base_link?.visual
    allFaces.push(
      ...createCylinderFaces(
        T_base,
        baseVis?.geometry?.radius ?? 0.3,
        baseVis?.geometry?.length ?? 0.2,
        baseVis?.origin_xyz ?? [0, 0, 0.1],
        baseVis?.color_rgba ?? [0.05, 0.3, 0.8, 1.0]
      )
    )

    // Link 1 box (size [0.18, 0.18, 1.0], origin [0, 0, 0.5])
    const l1Vis = links.link1?.visual
    allFaces.push(
      ...createBoxFaces(
        T_j1,
        l1Vis?.geometry?.size ?? [0.18, 0.18, 1.0],
        l1Vis?.origin_xyz ?? [0, 0, 0.5],
        l1Vis?.color_rgba ?? [0.1, 0.4, 0.9, 1.0]
      )
    )

    // Link 2 box (size [0.16, 0.16, 0.9], origin [0, 0, 0.45])
    const l2Vis = links.link2?.visual
    allFaces.push(
      ...createBoxFaces(
        T_j2,
        l2Vis?.geometry?.size ?? [0.16, 0.16, 0.9],
        l2Vis?.origin_xyz ?? [0, 0, 0.45],
        l2Vis?.color_rgba ?? [0.15, 0.65, 0.95, 1.0]
      )
    )

    // Link 3 box (size [0.14, 0.14, 0.8], origin [0, 0, 0.4])
    const l3Vis = links.link3?.visual
    allFaces.push(
      ...createBoxFaces(
        T_j3,
        l3Vis?.geometry?.size ?? [0.14, 0.14, 0.8],
        l3Vis?.origin_xyz ?? [0, 0, 0.4],
        l3Vis?.color_rgba ?? [0.95, 0.4, 0.05, 1.0]
      )
    )

    // End Effector box (size [0.25, 0.25, 0.24], origin [0, 0, 0.12])
    const eeVis = links.end_effector?.visual
    allFaces.push(
      ...createBoxFaces(
        T_ee,
        eeVis?.geometry?.size ?? [0.25, 0.25, 0.24],
        eeVis?.origin_xyz ?? [0, 0, 0.12],
        eeVis?.color_rgba ?? [0.8, 0.8, 0.82, 1.0]
      )
    )

    // 5. Painter's Algorithm Depth Sort + Directional Sun Shading
    const lightDir = normVec([0.5, 0.45, 0.85])
    const projectedFaces = []

    for (const face of allFaces) {
      const pts2d = []
      let sumDepth = 0
      let clipped = false
      for (const v of face.verts) {
        const p = project3D(v)
        if (!p) {
          clipped = true
          break
        }
        pts2d.push(p)
        sumDepth += p.depth
      }
      if (clipped || pts2d.length < 3) continue

      const e1 = subVec(face.verts[1], face.verts[0])
      const e2 = subVec(face.verts[2], face.verts[0])
      const normal = normVec(crossVec(e1, e2))
      const diffuse = Math.max(0.38, Math.min(1.08, 0.55 + 0.48 * Math.abs(dotVec(normal, lightDir))))

      const [r, g, b, a] = face.rgba
      const rr = Math.min(255, Math.round(r * 255 * diffuse))
      const gg = Math.min(255, Math.round(g * 255 * diffuse))
      const bb = Math.min(255, Math.round(b * 255 * diffuse))

      projectedFaces.push({
        pts2d,
        avgDepth: sumDepth / pts2d.length,
        fill: `rgba(${rr}, ${gg}, ${bb}, ${a ?? 1})`,
      })
    }

    projectedFaces.sort((a, b) => b.avgDepth - a.avgDepth)

    for (const pf of projectedFaces) {
      ctx.fillStyle = pf.fill
      ctx.strokeStyle = 'rgba(15, 23, 42, 0.55)'
      ctx.lineWidth = 0.8
      ctx.beginPath()
      pf.pts2d.forEach((p, idx) => {
        if (idx === 0) ctx.moveTo(p.x, p.y)
        else ctx.lineTo(p.x, p.y)
      })
      ctx.closePath()
      ctx.fill()
      ctx.stroke()
    }

    // 6. Draw Planned 3D Target Trajectory Waypoints (if processed demonstration exists)
    if (waypoints && waypoints.length > 0) {
      const wp2d = waypoints
        .map((wp, idx) => {
          const xyz = wp.target_xyz
          if (!xyz || xyz.length !== 3) return null
          const p = project3D(xyz)
          return p ? { ...p, step: wp.step ?? idx + 1 } : null
        })
        .filter(Boolean)

      if (wp2d.length > 1) {
        ctx.strokeStyle = 'rgba(250, 204, 21, 0.82)'
        ctx.lineWidth = 2.2
        ctx.setLineDash([4, 3])
        ctx.beginPath()
        wp2d.forEach((p, idx) => {
          if (idx === 0) ctx.moveTo(p.x, p.y)
          else ctx.lineTo(p.x, p.y)
        })
        ctx.stroke()
        ctx.setLineDash([])
      }

      wp2d.forEach((p, idx) => {
        const isCurrent = currentStep === p.step || (previewPlaying && previewIdx === idx)
        ctx.fillStyle = isCurrent ? '#f43f5e' : idx === 0 ? '#10b981' : '#facc15'
        ctx.beginPath()
        ctx.arc(p.x, p.y, isCurrent ? 5.5 : 3.2, 0, Math.PI * 2)
        ctx.fill()
      })
    }

    // 7. Draw Live Executed End-Effector Trail from Gazebo /joint_states
    if (executedTrail && executedTrail.length > 1) {
      const trail2d = executedTrail.map((xyz) => project3D(xyz)).filter(Boolean)
      if (trail2d.length > 1) {
        ctx.strokeStyle = '#10b981'
        ctx.lineWidth = 3
        ctx.beginPath()
        trail2d.forEach((p, idx) => {
          if (idx === 0) ctx.moveTo(p.x, p.y)
          else ctx.lineTo(p.x, p.y)
        })
        ctx.stroke()
      }
    }

    // 8. Draw Joint Pivot Markers & Live Angle Labels on the 3D Arm
    const j1World = transformPoint(T_j1, [0, 0, 0])
    const j2World = transformPoint(T_j2, [0, 0, 0])
    const j3World = transformPoint(T_j3, [0, 0, 0])
    const eeWorld = transformPoint(T_ee, [0, 0, 0.12])

    const jointMarkers = [
      { pt: j1World, label: `J1 (Yaw): ${q1.toFixed(2)} rad`, color: '#38bdf8' },
      { pt: j2World, label: `J2 (Shoulder): ${q2.toFixed(2)} rad`, color: '#a78bfa' },
      { pt: j3World, label: `J3 (Elbow): ${q3.toFixed(2)} rad`, color: '#fb923c' },
      {
        pt: eeWorld,
        label: `EE: (${eeWorld[0].toFixed(2)}, ${eeWorld[1].toFixed(2)}, ${eeWorld[2].toFixed(2)})m`,
        color: '#34d399',
      },
    ]

    jointMarkers.forEach((jm) => {
      const p = project3D(jm.pt)
      if (!p) return
      ctx.fillStyle = jm.color
      ctx.beginPath()
      ctx.arc(p.x, p.y, 4.5, 0, Math.PI * 2)
      ctx.fill()
      ctx.strokeStyle = '#0f172a'
      ctx.lineWidth = 1.5
      ctx.stroke()

      ctx.fillStyle = 'rgba(15, 23, 42, 0.82)'
      const textW = ctx.measureText(jm.label).width + 10
      ctx.fillRect(p.x + 8, p.y - 10, textW, 16)
      ctx.fillStyle = jm.color
      ctx.font = '10px JetBrains Mono, monospace'
      ctx.fillText(jm.label, p.x + 12, p.y + 1)
    })
  }, [
    urdfSpec,
    activeJoints,
    camOrbit,
    waypoints,
    executedTrail,
    currentStep,
    previewPlaying,
    previewIdx,
  ])

  const handleMouseDown = (e) => {
    dragRef.current = { active: true, lastX: e.clientX, lastY: e.clientY }
  }

  const handleMouseMove = (e) => {
    if (!dragRef.current.active) return
    const dx = e.clientX - dragRef.current.lastX
    const dy = e.clientY - dragRef.current.lastY
    dragRef.current.lastX = e.clientX
    dragRef.current.lastY = e.clientY
    setCamOrbit((prev) => ({
      ...prev,
      azimuth: prev.azimuth - dx * 0.008,
      elevation: Math.max(-0.2, Math.min(1.35, prev.elevation + dy * 0.006)),
    }))
  }

  const handleMouseUp = () => {
    dragRef.current.active = false
  }

  const handleWheel = (e) => {
    e.preventDefault()
    setCamOrbit((prev) => ({
      ...prev,
      distance: Math.max(2.2, Math.min(7.5, prev.distance + e.deltaY * 0.002)),
    }))
  }

  return (
    <div className="robot-viewport-wrapper">
      <div className="viewport-overlay-bar">
        <div className="overlay-badge">
          <span
            className="status-dot"
            style={{
              background: previewPlaying
                ? '#f59e0b'
                : isGazeboRunning && isControllerReady
                ? '#10b981'
                : '#ef4444',
            }}
          />
          <span>
            {previewPlaying
              ? `3D TRAJECTORY PREVIEW (Step ${previewIdx + 1}/${waypoints.length})`
              : isGazeboRunning && isControllerReady
              ? `LIVE 3D URDF VIEWPORT • Synced to Gazebo /joint_states (${
                  wsConnected ? 'WebSocket 25Hz' : 'HTTP Poll'
                })`
              : '3D URDF MODEL VIEW • Gazebo Offline (Showing Home Pose)'}
          </span>
        </div>

        <div className="viewport-camera-btns">
          {waypoints.length > 0 && (
            <button
              className="btn-mini"
              onClick={() => {
                setPreviewIdx(0)
                setPreviewPlaying((p) => !p)
              }}
              disabled={executionStatus === 'executing'}
              title="Preview the generated robot trajectory in the 3D viewport before executing in Gazebo"
            >
              {previewPlaying ? '■ Stop Preview' : '👁 Preview Motion'}
            </button>
          )}
          <button
            className="btn-mini"
            onClick={() =>
              setCamOrbit({ azimuth: 0.68, elevation: 0.36, distance: 4.15, targetZ: 1.32 })
            }
          >
            Iso
          </button>
          <button
            className="btn-mini"
            onClick={() =>
              setCamOrbit({ azimuth: 0.0, elevation: 0.15, distance: 4.2, targetZ: 1.32 })
            }
          >
            Front
          </button>
          <button
            className="btn-mini"
            onClick={() =>
              setCamOrbit({ azimuth: 1.57, elevation: 0.15, distance: 4.2, targetZ: 1.32 })
            }
          >
            Side
          </button>
          <button
            className="btn-mini"
            onClick={() =>
              setCamOrbit({ azimuth: 0.0, elevation: 1.3, distance: 4.6, targetZ: 1.32 })
            }
          >
            Top
          </button>
        </div>
      </div>

      <canvas
        ref={canvasRef}
        width={680}
        height={430}
        className="robot-viewport-canvas"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
      />

      <div className="viewport-overlay-bottom">
        <span>
          Model: <code>human_robot_arm.urdf</code> (L1=0.9m, L2=0.8m, Base Z=1.2m)
        </span>
        <span>
          <span style={{ color: '#facc15' }}>--- Planned Path</span> •{' '}
          <span style={{ color: '#10b981' }}>━━ Gazebo Executed Path</span> • Drag to Orbit / Scroll
          to Zoom
        </span>
      </div>
    </div>
  )
}

