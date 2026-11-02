import React, { useEffect, useRef } from 'react'

/**
 * Renders a 2D visualization of the captured human hand trajectory over time
 * (normalized image X vs Y, with Z depth encoded by point radius/color gradient).
 * Used in the Left Panel (Human Demonstration) both live during recording and
 * after recording/processing completes.
 */
export default function HandTrajectoryCanvas({
  livePoints = [],
  processedPoints = [],
  isRecording = false,
  latestWrist = null,
}) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const w = canvas.width
    const h = canvas.height

    ctx.clearRect(0, 0, w, h)

    // Background & subtle coordinate grid
    ctx.fillStyle = '#070b14'
    ctx.fillRect(0, 0, w, h)

    ctx.strokeStyle = 'rgba(148, 163, 184, 0.12)'
    ctx.lineWidth = 1
    for (let i = 1; i < 4; i++) {
      const gx = (w / 4) * i
      const gy = (h / 4) * i
      ctx.beginPath()
      ctx.moveTo(gx, 0)
      ctx.lineTo(gx, h)
      ctx.stroke()

      ctx.beginPath()
      ctx.moveTo(0, gy)
      ctx.lineTo(w, gy)
      ctx.stroke()
    }

    // Choose active point source: live recording points while recording, or processed/saved points
    const points =
      isRecording && livePoints.length > 0
        ? livePoints.map((p) => ({
            x: p.wrist?.[0] ?? 0.5,
            y: p.wrist?.[1] ?? 0.5,
            z: p.wrist?.[2] ?? 0.0,
            palm: p.palm_center,
          }))
        : processedPoints.length > 0
        ? processedPoints.map((p) => ({
            x: p.smoothed_wrist?.[0] ?? p.raw_wrist?.[0] ?? 0.5,
            y: p.smoothed_wrist?.[1] ?? p.raw_wrist?.[1] ?? 0.5,
            z: p.smoothed_wrist?.[2] ?? p.raw_wrist?.[2] ?? 0.0,
            raw: p.raw_wrist,
            palm: p.palm_center,
          }))
        : livePoints.map((p) => ({
            x: p.wrist?.[0] ?? 0.5,
            y: p.wrist?.[1] ?? 0.5,
            z: p.wrist?.[2] ?? 0.0,
            palm: p.palm_center,
          }))

    const pad = 22
    const toPx = (nx, ny) => [
      pad + Math.max(0, Math.min(1, nx)) * (w - 2 * pad),
      pad + Math.max(0, Math.min(1, ny)) * (h - 2 * pad),
    ]

    if (points.length === 0) {
      ctx.fillStyle = '#64748b'
      ctx.font = '12px Inter, sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText(
        isRecording
          ? 'Waiting for hand landmarks in camera view...'
          : 'Captured hand trajectory path will appear here during & after recording.',
        w / 2,
        h / 2
      )
    } else {
      // Draw raw noisy wrist trace faintly if available
      if (processedPoints.length > 1 && points[0].raw) {
        ctx.strokeStyle = 'rgba(148, 163, 184, 0.32)'
        ctx.lineWidth = 1.2
        ctx.setLineDash([3, 3])
        ctx.beginPath()
        points.forEach((pt, idx) => {
          if (!pt.raw) return
          const [px, py] = toPx(pt.raw[0], pt.raw[1])
          if (idx === 0) ctx.moveTo(px, py)
          else ctx.lineTo(px, py)
        })
        ctx.stroke()
        ctx.setLineDash([])
      }

      // Draw palm-centre path in subtle cyan
      if (points.length > 1 && points[0].palm) {
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.45)'
        ctx.lineWidth = 1.5
        ctx.beginPath()
        points.forEach((pt, idx) => {
          if (!pt.palm) return
          const [px, py] = toPx(pt.palm[0], pt.palm[1])
          if (idx === 0) ctx.moveTo(px, py)
          else ctx.lineTo(px, py)
        })
        ctx.stroke()
      }

      // Draw primary smoothed wrist trajectory path
      if (points.length > 1) {
        ctx.strokeStyle = isRecording ? '#f43f5e' : '#facc15'
        ctx.lineWidth = 2.5
        ctx.beginPath()
        points.forEach((pt, idx) => {
          const [px, py] = toPx(pt.x, pt.y)
          if (idx === 0) ctx.moveTo(px, py)
          else ctx.lineTo(px, py)
        })
        ctx.stroke()
      }

      // Draw sampled points along trajectory
      const step = Math.max(1, Math.floor(points.length / 25))
      points.forEach((pt, idx) => {
        if (idx % step !== 0 && idx !== points.length - 1) return
        const [px, py] = toPx(pt.x, pt.y)
        ctx.fillStyle = idx === 0 ? '#10b981' : idx === points.length - 1 ? '#ef4444' : '#38bdf8'
        ctx.beginPath()
        ctx.arc(px, py, idx === 0 || idx === points.length - 1 ? 4.5 : 2.5, 0, Math.PI * 2)
        ctx.fill()
      })

      // Label Start and End points
      const [sx, sy] = toPx(points[0].x, points[0].y)
      ctx.fillStyle = '#10b981'
      ctx.font = 'bold 10px JetBrains Mono, monospace'
      ctx.textAlign = 'left'
      ctx.fillText('START', sx + 7, sy + 3)

      if (points.length > 1) {
        const last = points[points.length - 1]
        const [ex, ey] = toPx(last.x, last.y)
        ctx.fillStyle = '#f87171'
        ctx.fillText('END', ex + 7, ey + 3)
      }
    }

    // Draw live current wrist cursor if visible right now
    if (latestWrist && latestWrist.length === 3) {
      const [lx, ly] = toPx(latestWrist[0], latestWrist[1])
      ctx.strokeStyle = '#34d399'
      ctx.lineWidth = 1.5
      ctx.beginPath()
      ctx.arc(lx, ly, 7, 0, Math.PI * 2)
      ctx.stroke()
    }

    // Legend in top-left & bottom-right
    ctx.fillStyle = '#94a3b8'
    ctx.font = '10px JetBrains Mono, monospace'
    ctx.textAlign = 'left'
    ctx.fillText(
      `Hand Trajectory Map (Wrist #0 & Palm Centre) • ${points.length} valid samples`,
      8,
      14
    )
    ctx.textAlign = 'right'
    ctx.fillText('Camera Frame [0..1] → Robot Workspace YZ', w - 8, h - 6)
  }, [livePoints, processedPoints, isRecording, latestWrist])

  return (
    <div className="trajectory-canvas-wrap">
      <canvas
        ref={canvasRef}
        width={540}
        height={135}
        className="trajectory-canvas"
      />
    </div>
  )
}

