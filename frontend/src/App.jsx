import React, { useState, useEffect, useCallback, useRef } from 'react'
import HandTrajectoryCanvas from './HandTrajectoryCanvas'
import RobotViewport3D from './RobotViewport3D'

function formatDuration(seconds) {
  if (typeof seconds !== 'number' || Number.isNaN(seconds)) return '0.0s'
  const totalSec = Math.max(0, Math.floor(seconds))
  const hrs = Math.floor(totalSec / 3600)
  const mins = Math.floor((totalSec % 3600) / 60)
  const secs = totalSec % 60
  if (hrs > 0) {
    return `${hrs}h ${String(mins).padStart(2, '0')}m ${String(secs).padStart(2, '0')}s`
  }
  return `${String(mins).padStart(2, '0')}m ${String(secs).padStart(2, '0')}s`
}

function jointPercent(angleRad, lower, upper) {
  if (typeof angleRad !== 'number' || Number.isNaN(angleRad)) return 50
  const span = upper - lower
  if (span <= 0) return 50
  const clamped = Math.max(lower, Math.min(upper, angleRad))
  return Math.round(((clamped - lower) / span) * 100)
}

const DEFAULT_PIPELINE_STAGES = [
  { id: 'human_hand_trajectory', label: '1. Human Hand Trajectory', status: 'idle', detail: 'Record or select a demonstration' },
  { id: 'validate_recorded_landmarks', label: '2. Validate Recorded Landmarks', status: 'idle', detail: '21 3D landmarks check' },
  { id: 'normalize_and_smooth', label: '3. Normalize & Smooth Trajectory', status: 'idle', detail: 'Wrist & palm reference + EMA' },
  { id: 'extract_motion_features', label: '4. Extract Motion Features', status: 'idle', detail: '3D displacement vectors' },
  { id: 'convert_robot_targets', label: '5. Convert to Robot-Reachable Targets', status: 'idle', detail: 'BC model + workspace mapping' },
  { id: 'generate_joint_trajectory', label: '6. Generate Robot Joint Trajectory', status: 'idle', detail: 'Analytical 3-DOF IK & FK' },
  { id: 'validate_joint_limits', label: '7. Validate Robot Joint Limits', status: 'idle', detail: 'URDF limits & continuity' },
  { id: 'preview_robot_motion', label: '8. Preview & Execute in Gazebo', status: 'idle', detail: '3D preview & ROS 2 execution' },
]

export default function App() {
  const [health, setHealth] = useState(null)
  const [connectionState, setConnectionState] = useState('loading') // loading | online | offline
  const [fetchError, setFetchError] = useState(null)
  const [lastUpdated, setLastUpdated] = useState(null)

  // URDF Spec & Live WebSocket Robot State
  const [urdfSpec, setUrdfSpec] = useState(null)
  const [wsRobotState, setWsRobotState] = useState(null)
  const [wsConnected, setWsConnected] = useState(false)

  // Stream state
  const [streamKey, setStreamKey] = useState(1)
  const [streamLoaded, setStreamLoaded] = useState(false)
  const [streamError, setStreamError] = useState(false)

  // Demonstrations & Processing state
  const [demonstrations, setDemonstrations] = useState([])
  const [selectedDemoId, setSelectedDemoId] = useState('')
  const [selectedDemoDetail, setSelectedDemoDetail] = useState(null)
  const [autoExecute, setAutoExecute] = useState(false)
  const [showWaypointsTable, setShowWaypointsTable] = useState(false)

  // UI Action Feedback
  const [actionBanner, setActionBanner] = useState(null)
  const [busyAction, setBusyAction] = useState(null)

  const isPollingRef = useRef(false)
  const selectedDemoIdRef = useRef('')
  selectedDemoIdRef.current = selectedDemoId

  // Fetch URDF Specification once on mount
  const fetchUrdfSpec = useCallback(async () => {
    try {
      const res = await fetch('/api/robot/urdf')
      if (!res.ok) return
      const data = await res.json()
      if (data.spec) {
        setUrdfSpec(data.spec)
      }
    } catch {
      // Handled by primary health poll
    }
  }, [])

  // Fetch Demonstrations List (prefer valid processed demos on initial load)
  const fetchDemonstrationsList = useCallback(async (autoSelectId = null) => {
    try {
      const res = await fetch('/api/demonstrations')
      if (!res.ok) return
      const data = await res.json()
      const list = data.demonstrations || []
      setDemonstrations(list)

      let targetId = autoSelectId || selectedDemoIdRef.current
      if (!targetId && list.length > 0) {
        const firstValid = list.find(
          (d) => (d.detected_hand_frames ?? 0) > 0 && d.processing_status !== 'rejected'
        )
        targetId = (firstValid || list[0]).demonstration_id
      }

      if (targetId) {
        setSelectedDemoId(targetId)
        const found = list.find((d) => d.demonstration_id === targetId)
        if (found) {
          setSelectedDemoDetail(found)
        }
      }
    } catch {
      // Handled by primary health poll
    }
  }, [])

  const pollHealth = useCallback(async () => {
    if (isPollingRef.current) return
    isPollingRef.current = true

    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 3000)

    try {
      const res = await fetch('/api/health', {
        signal: controller.signal,
        headers: { Accept: 'application/json' },
      })
      clearTimeout(timeoutId)

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`)
      }

      const data = await res.json()
      setHealth(data)
      setConnectionState('online')
      setFetchError(null)
      setLastUpdated(new Date().toLocaleTimeString())
    } catch (err) {
      clearTimeout(timeoutId)
      setConnectionState('offline')
      setFetchError(
        err.name === 'AbortError'
          ? 'Backend poll timed out (3s)'
          : err.message || 'Unable to reach FastAPI backend at /api/health'
      )
    } finally {
      isPollingRef.current = false
    }
  }, [])

  useEffect(() => {
    fetchUrdfSpec()
    pollHealth()
    fetchDemonstrationsList()
    const interval = setInterval(pollHealth, 750)
    return () => clearInterval(interval)
  }, [fetchUrdfSpec, pollHealth, fetchDemonstrationsList])

  // Real-time WebSocket connection for 25 Hz /joint_states streaming
  useEffect(() => {
    let ws = null
    let reconnectTimer = null
    let unmounted = false

    const connectWs = () => {
      if (unmounted) return
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const wsUrl = `${proto}//${window.location.host}/api/ws/robot-state`
      try {
        ws = new WebSocket(wsUrl)
        ws.onopen = () => {
          if (!unmounted) setWsConnected(true)
        }
        ws.onmessage = (event) => {
          if (unmounted) return
          try {
            const payload = JSON.parse(event.data)
            setWsRobotState(payload)
          } catch {
            // Ignore malformed frame
          }
        }
        ws.onclose = () => {
          if (!unmounted) {
            setWsConnected(false)
            reconnectTimer = setTimeout(connectWs, 2000)
          }
        }
        ws.onerror = () => {
          try {
            ws.close()
          } catch {
            // Ignore
          }
        }
      } catch {
        reconnectTimer = setTimeout(connectWs, 2000)
      }
    }

    connectWs()
    return () => {
      unmounted = true
      if (reconnectTimer) clearTimeout(reconnectTimer)
      if (ws) {
        try {
          ws.close()
        } catch {
          // Ignore
        }
      }
    }
  }, [])

  useEffect(() => {
    if (!selectedDemoId) {
      setSelectedDemoDetail(null)
      return
    }
    const found = demonstrations.find((d) => d.demonstration_id === selectedDemoId)
    if (found) {
      setSelectedDemoDetail(found)
    }
  }, [selectedDemoId, demonstrations])

  const handleReloadStream = () => {
    setStreamLoaded(false)
    setStreamError(false)
    setStreamKey((prev) => prev + 1)
  }

  // API Action Helper
  const runApiAction = async (label, url, options = {}, onSuccess = null) => {
    setBusyAction(label)
    setActionBanner(null)
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        ...options,
      })
      const payload = await res.json().catch(() => ({}))
      if (!res.ok) {
        throw new Error(payload.detail || payload.message || `HTTP ${res.status}`)
      }
      if (onSuccess) {
        await onSuccess(payload)
      }
      await pollHealth()
    } catch (err) {
      setActionBanner({ type: 'error', text: `${label}: ${err.message}` })
      await pollHealth()
    } finally {
      setBusyAction(null)
    }
  }

  const handleStartCamera = () =>
    runApiAction('Start Camera', '/api/camera/start', {}, async () => {
      handleReloadStream()
      setActionBanner({ type: 'info', text: 'Camera capture started on /dev/video0.' })
    })

  const handleStopCamera = () =>
    runApiAction('Stop Camera', '/api/camera/stop', {}, async (payload) => {
      setStreamLoaded(false)
      if (payload.saved_active_recording) {
        const savedId = payload.saved_active_recording.demonstration_id
        await fetchDemonstrationsList(savedId)
        setActionBanner({
          type: 'info',
          text: `Active recording '${savedId}' saved and camera stopped safely.`,
        })
      } else {
        setActionBanner({ type: 'info', text: 'Camera stopped and /dev/video0 released.' })
      }
    })

  const handleStartRecording = () =>
    runApiAction('Start Recording', '/api/recording/start', {}, async (payload) => {
      setActionBanner({
        type: 'info',
        text: `Recording started (${payload.demonstration_id}). Move your hand smoothly inside the camera frame.`,
      })
    })

  const handleStopRecording = () =>
    runApiAction(
      'Stop & Process Recording',
      '/api/recording/stop',
      {
        body: JSON.stringify({
          auto_process: true,
          auto_execute: autoExecute,
        }),
      },
      async (payload) => {
        const newId = payload.demonstration_id
        await fetchDemonstrationsList(newId)
        const summary = payload.processing_summary
        if (summary && summary.status === 'processed') {
          const autoMsg = payload.auto_execution
            ? ' Auto-executing on robot in Gazebo now!'
            : ' Click "Execute Demonstration" in the right panel to run on the robot.'
          setActionBanner({
            type: 'info',
            text: `Recording '${newId}' complete (${payload.frame_count} frames, ${payload.detected_hand_frames} valid hand frames) → Validated ${summary.waypoint_count} robot waypoints.${autoMsg}`,
          })
        } else {
          setActionBanner({
            type: 'warning',
            text: `Saved '${newId}', but processing reported: ${summary?.error || 'Check status'}`,
          })
        }
      }
    )

  const handleProcessDemo = (demoIdToProcess = selectedDemoId) => {
    if (!demoIdToProcess) return
    runApiAction(
      'Process Trajectory',
      `/api/demonstrations/${encodeURIComponent(demoIdToProcess)}/process`,
      {},
      async (summary) => {
        await fetchDemonstrationsList(demoIdToProcess)
        if (summary.status === 'processed') {
          setActionBanner({
            type: 'info',
            text: `Processed '${demoIdToProcess}': ${summary.waypoint_count} URDF-validated robot waypoints ready for execution.`,
          })
        } else {
          setActionBanner({
            type: 'error',
            text: `Trajectory '${demoIdToProcess}' rejected: ${summary.error}`,
          })
        }
      }
    )
  }

  const handleConnectSim = () =>
    runApiAction('Refresh Simulation', '/api/robot/connect', {}, async (payload) => {
      setActionBanner({
        type: payload.controller_available ? 'info' : 'warning',
        text: payload.controller_available
          ? 'Verified ROS 2 Jazzy, Gazebo Sim 8, and arm_controller are online and ready.'
          : payload.status?.status_reason || 'Simulation or controller not yet active.',
      })
    })

  const handleLaunchSim = () =>
    runApiAction('Launch Gazebo Simulation', '/api/robot/launch-sim', {}, async (payload) => {
      setActionBanner({
        type: 'info',
        text: payload.message || 'Launched robot_sim.launch.py.',
      })
    })

  const handleTestMotion = () =>
    runApiAction('Test Robot Motion', '/api/robot/test-motion', {}, async (payload) => {
      setActionBanner({
        type: 'info',
        text: payload.message || 'Executing 3-waypoint test trajectory in Gazebo.',
      })
    })

  const handleExecuteDemo = () => {
    if (!selectedDemoId) return
    runApiAction(
      'Execute Demonstration',
      '/api/robot/execute',
      { body: JSON.stringify({ demonstration_id: selectedDemoId }) },
      async (payload) => {
        setActionBanner({
          type: 'info',
          text: `Executing '${payload.demonstration_id}' (${payload.waypoint_count} waypoints) on simulated robot arm.`,
        })
      }
    )
  }

  const handlePauseResume = (isPaused) => {
    const endpoint = isPaused ? '/api/robot/resume' : '/api/robot/pause'
    const label = isPaused ? 'Resume Execution' : 'Pause Execution'
    runApiAction(label, endpoint, {}, async (payload) => {
      setActionBanner({
        type: 'info',
        text: payload.message,
      })
    })
  }

  const handleResetRobot = () =>
    runApiAction('Reset Simulation', '/api/robot/reset', {}, async (payload) => {
      setActionBanner({
        type: 'info',
        text: payload.message || 'Robot arm reset to Home pose [0.0, 0.0, 0.0] rad.',
      })
    })

  const handleStopRobot = () =>
    runApiAction('Stop Robot', '/api/robot/stop', {}, async (payload) => {
      setActionBanner({
        type: 'warning',
        text: payload.message || 'Robot motion stopped safely.',
      })
    })

  // Derived real state
  const isBackendOnline = connectionState === 'online' && health?.status === 'online'
  const camera = health?.camera || {}
  const recording = health?.recording || {}
  const modelKin = health?.model_kinematics || {}
  const robot = { ...(health?.robot || {}), ...(wsRobotState || {}) }

  const isCameraConnected = Boolean(isBackendOnline && camera.connected)
  const isRecording = Boolean(isBackendOnline && (camera.is_recording || recording.is_recording))
  const isHandDetected = Boolean(isCameraConnected && camera.hand_detected)

  const isRos2Connected = Boolean(isBackendOnline && robot.ros2_connected)
  const isGazeboRunning = Boolean(isBackendOnline && robot.gazebo_running)
  const isControllerReady = Boolean(isBackendOnline && robot.controller_available)

  const rawExecStatus = robot.command_execution_status || 'ready'
  const isPaused = Boolean(robot.is_paused || rawExecStatus === 'paused')
  const isRobotExecuting =
    isBackendOnline &&
    (rawExecStatus === 'executing' ||
      rawExecStatus === 'playing' ||
      rawExecStatus === 'testing' ||
      rawExecStatus === 'paused')

  // Recording status badge: Idle | Recording | Processing | Complete | Error
  const recStateRaw = recording.recording_state || camera.recording_state || 'idle'
  const recordingStatusLabel = isRecording
    ? 'Recording'
    : busyAction === 'Stop & Process Recording' || busyAction === 'Process Trajectory' || recStateRaw === 'processing'
    ? 'Processing'
    : recStateRaw === 'error'
    ? 'Error'
    : recStateRaw === 'complete' || selectedDemoDetail?.processing_status === 'processed'
    ? 'Complete'
    : 'Idle'

  // Execution state badge: Ready | Validating | Executing | Paused | Completed | Stopped | Error
  const executionStateLabel = !isGazeboRunning || !isControllerReady
    ? 'Sim Offline'
    : rawExecStatus === 'validating'
    ? 'Validating'
    : isPaused
    ? 'Paused'
    : rawExecStatus === 'executing' || rawExecStatus === 'playing' || rawExecStatus === 'testing'
    ? 'Executing'
    : rawExecStatus === 'completed'
    ? 'Completed'
    : rawExecStatus === 'stopped'
    ? 'Stopped'
    : rawExecStatus === 'error'
    ? 'Error'
    : 'Ready'

  const procSummary = selectedDemoDetail?.processing_summary || null
  const isDemoProcessed = Boolean(procSummary && procSummary.status === 'processed')
  const waypoints = procSummary?.waypoints || []
  const pipelineStages = procSummary?.pipeline_stages || DEFAULT_PIPELINE_STAGES

  const jointNames = health?.robot?.joint_names || ['joint1', 'joint2', 'joint3']
  const jointLimits = health?.robot?.joint_limits || {
    joint1: { lower: -2.5, upper: 2.5 },
    joint2: { lower: -1.57, upper: 1.57 },
    joint3: { lower: -1.57, upper: 1.57 },
  }
  const actualJointsRad =
    robot.positions_rad || health?.robot?.latest_joint_positions_rad || null
  const actualJointsDeg =
    robot.positions_deg || health?.robot?.latest_joint_positions_deg || null
  const targetJointsRad =
    robot.targets_rad || health?.robot?.latest_joint_targets_rad || null
  const executedTrail = robot.executed_ee_trail || health?.robot?.executed_ee_trail || []

  // Displayed frame counters in Left Panel
  const shownTotalFrames = isRecording
    ? camera.recording_frames ?? recording.recorded_frames ?? 0
    : selectedDemoDetail?.frame_count ?? recording.recorded_frames ?? 0
  const shownValidHandFrames = isRecording
    ? camera.recording_valid_frames ?? recording.detected_hand_frames ?? 0
    : selectedDemoDetail?.detected_hand_frames ?? recording.detected_hand_frames ?? 0
  const shownDurationSec = isRecording
    ? camera.recording_duration_seconds ?? recording.recording_duration_seconds ?? 0
    : selectedDemoDetail?.duration_seconds ?? recording.recording_duration_seconds ?? 0

  return (
    <div className="dashboard-container">
      {/* TOP BAR: PROJECT TITLE & ACTUAL SYSTEM CONNECTION INDICATORS */}
      <header className="dashboard-header">
        <div className="header-brand">
          <div className="brand-icon">RC</div>
          <div className="brand-title">
            <h1>AI-Powered Robot Arm Learning from Human Demonstration</h1>
            <div className="brand-subtitle">
              <span className="stage-badge">Live Human-to-Robot Teleoperation &amp; BC Console</span>
              <span>MediaPipe 21-Keypoint Vision • PyTorch BC • Analytical IK • ROS 2 Jazzy • Gazebo Sim 8</span>
              {isBackendOnline && (
                <span>• Uptime: {formatDuration(health?.uptime_seconds)}</span>
              )}
            </div>
          </div>
        </div>

        <div className="header-status-bar">
          <div className={`status-pill ${isBackendOnline ? 'online' : 'offline'}`}>
            <span className="status-dot" />
            <span>API: {isBackendOnline ? 'ONLINE' : 'OFFLINE'}</span>
          </div>

          <div className={`status-pill ${isCameraConnected ? 'online' : 'offline'}`}>
            <span className="status-dot" />
            <span>CAMERA: {isCameraConnected ? 'ACTIVE' : 'OFF'}</span>
          </div>

          <div
            className={`status-pill ${
              !isCameraConnected ? 'offline' : isHandDetected ? 'online' : 'loading'
            }`}
          >
            <span className="status-dot" />
            <span>
              HAND: {!isCameraConnected ? 'OFF' : isHandDetected ? '21 LANDMARKS' : 'NO HAND'}
            </span>
          </div>

          <div className={`status-pill ${modelKin.model_loaded ? 'online' : 'offline'}`}>
            <span className="status-dot" />
            <span>BC MODEL: {modelKin.model_loaded ? 'LOADED' : 'ERROR'}</span>
          </div>

          <div className={`status-pill ${isRos2Connected ? 'online' : 'offline'}`}>
            <span className="status-dot" />
            <span>ROS 2: {isRos2Connected ? 'CONNECTED' : 'OFFLINE'}</span>
          </div>

          <div
            className={`status-pill ${
              isGazeboRunning && isControllerReady
                ? 'online'
                : isGazeboRunning
                ? 'loading'
                : 'offline'
            }`}
          >
            <span className="status-dot" />
            <span>
              GAZEBO: {isGazeboRunning && isControllerReady ? 'SIM READY' : 'OFFLINE'}
            </span>
          </div>
        </div>
      </header>

      {/* BANNERS */}
      {connectionState === 'offline' && (
        <div className="alert-banner error" role="alert">
          <div>
            <strong>Backend Unreachable:</strong> {fetchError}. Start FastAPI on{' '}
            <code>http://127.0.0.1:8000</code>.
          </div>
          <button className="btn" onClick={pollHealth}>
            Retry
          </button>
        </div>
      )}

      {actionBanner && (
        <div className={`alert-banner ${actionBanner.type}`} role="status">
          <div>{actionBanner.text}</div>
          <button className="btn" onClick={() => setActionBanner(null)}>
            Dismiss
          </button>
        </div>
      )}

      {/* =================================================================== */}
      {/* TWO LARGE SIDE-BY-SIDE PANELS: LEFT (HUMAN DEMO) | RIGHT (ROBOT SIM) */}
      {/* =================================================================== */}
      <main className="two-panel-grid">
        {/* LEFT PANEL — HUMAN DEMONSTRATION */}
        <section className="panel main-split-panel">
          <div className="panel-header">
            <div className="panel-title">
              LEFT PANEL — HUMAN DEMONSTRATION (Webcam &amp; MediaPipe 21 Landmarks)
            </div>
            <div className="panel-actions">
              <span
                className={`state-tag ${recordingStatusLabel.toLowerCase()}`}
                title="Current Recording Pipeline Status"
              >
                Status: {recordingStatusLabel.toUpperCase()}
              </span>
              <button
                className="btn success"
                onClick={handleStartCamera}
                disabled={!isBackendOnline || isCameraConnected || busyAction !== null}
              >
                Start Camera
              </button>
              <button
                className="btn danger"
                onClick={handleStopCamera}
                disabled={!isBackendOnline || !isCameraConnected || busyAction !== null}
              >
                Stop Camera
              </button>
              <button
                className="btn"
                onClick={handleReloadStream}
                disabled={!isCameraConnected}
                title="Reload MJPEG feed"
              >
                Reload
              </button>
            </div>
          </div>

          {/* LIVE WEBCAM STREAM VIEWPORT */}
          <div className="video-viewport">
            {isCameraConnected && !streamError ? (
              <>
                <img
                  key={streamKey}
                  src={`/stream/webcam?t=${streamKey}`}
                  alt="Live Webcam with MediaPipe Hand Landmarks"
                  className="video-stream-img"
                  onLoad={() => {
                    setStreamLoaded(true)
                    setStreamError(false)
                  }}
                  onError={() => {
                    setStreamLoaded(false)
                    setStreamError(true)
                  }}
                />
                <div className="video-overlay-top">
                  <div className="overlay-badge">
                    <span
                      className="status-dot"
                      style={{
                        background: isRecording
                          ? '#ef4444'
                          : streamLoaded
                          ? '#10b981'
                          : '#f59e0b',
                      }}
                    />
                    <span>
                      {isRecording
                        ? `RECORDING • ${shownDurationSec.toFixed(1)}s (${shownTotalFrames}f / ${shownValidHandFrames} valid)`
                        : streamLoaded
                        ? `LIVE • /dev/video${camera.device_id ?? 0}`
                        : 'BUFFERING...'}
                    </span>
                  </div>

                  <div className="overlay-badge">
                    <span>
                      {typeof camera.fps === 'number' ? `${camera.fps.toFixed(1)} FPS` : '— FPS'}
                    </span>
                    <span>•</span>
                    <span style={{ color: isHandDetected ? '#34d399' : '#fbbf24' }}>
                      {isHandDetected ? '21 Landmarks Detected' : 'Show Hand in Frame'}
                    </span>
                  </div>
                </div>
              </>
            ) : (
              <div className="video-placeholder">
                {connectionState === 'loading' ? (
                  <>
                    <h3>Connecting to Vision Pipeline...</h3>
                    <p>Checking /api/health and /stream/webcam</p>
                  </>
                ) : !isBackendOnline ? (
                  <>
                    <h3>Backend Offline</h3>
                    <p>Start the FastAPI backend to enable webcam capture and hand tracking.</p>
                  </>
                ) : (
                  <>
                    <h3>Camera Stopped</h3>
                    <p>
                      {camera.error ||
                        'Camera /dev/video0 is released. Click "Start Camera" to begin live hand tracking.'}
                    </p>
                    <button className="btn success" onClick={handleStartCamera}>
                      Start Camera
                    </button>
                  </>
                )}
              </div>
            )}
          </div>

          {/* RECORDING ACTION CONTROLS */}
          <div className="recording-controls-bar">
            <button
              className="btn primary btn-lg"
              onClick={handleStartRecording}
              disabled={!isCameraConnected || isRecording || busyAction !== null}
            >
              ● Start Recording
            </button>
            <button
              className="btn danger btn-lg"
              onClick={handleStopRecording}
              disabled={!isRecording || busyAction !== null}
            >
              ■ Stop Recording &amp; Process
            </button>
            <label
              className="auto-exec-toggle"
              title="When enabled, automatically executes the trajectory on the simulated robot right after validation succeeds"
            >
              <input
                type="checkbox"
                checked={autoExecute}
                onChange={(e) => setAutoExecute(e.target.checked)}
              />
              <span>Auto-Execute on Robot after Validation</span>
            </label>
          </div>

          {/* RECORDING TELEMETRY STRIP */}
          <div className="metrics-grid four-cols">
            <div className="metric-card">
              <span className="metric-label">Recording Status</span>
              <span
                className={`metric-value ${
                  recordingStatusLabel === 'Recording'
                    ? 'red'
                    : recordingStatusLabel === 'Error'
                    ? 'red'
                    : recordingStatusLabel === 'Complete'
                    ? 'emerald'
                    : 'cyan'
                }`}
              >
                {recordingStatusLabel.toUpperCase()}
              </span>
              <span className="metric-sub">
                ID: {isRecording ? recording.current_demonstration_id : selectedDemoId || 'None'}
              </span>
            </div>

            <div className="metric-card">
              <span className="metric-label">Duration &amp; Frames</span>
              <span className="metric-value">
                {Number(shownDurationSec).toFixed(2)}s / {shownTotalFrames}f
              </span>
              <span className="metric-sub">
                Camera: {typeof camera.fps === 'number' ? `${camera.fps.toFixed(1)} FPS` : '—'}
              </span>
            </div>

            <div className="metric-card">
              <span className="metric-label">Valid Hand Frames</span>
              <span className={`metric-value ${shownValidHandFrames > 0 ? 'emerald' : 'amber'}`}>
                {shownValidHandFrames} / {shownTotalFrames}
              </span>
              <span className="metric-sub">
                {isHandDetected && camera.latest_wrist
                  ? `Wrist: (${camera.latest_wrist[0].toFixed(2)}, ${camera.latest_wrist[1].toFixed(2)})`
                  : '21-keypoint detector'}
              </span>
            </div>

            <div className="metric-card">
              <span className="metric-label">Saved Output File</span>
              <span className="metric-value cyan" style={{ fontSize: '0.82rem' }}>
                {selectedDemoDetail ? 'landmarks.csv' : '—'}
              </span>
              <span className="metric-sub">
                {selectedDemoDetail
                  ? `data/demonstrations/${selectedDemoDetail.demonstration_id}/`
                  : 'No file selected'}
              </span>
            </div>
          </div>

          {/* CAPTURED HAND TRAJECTORY VISUALIZATION */}
          <HandTrajectoryCanvas
            livePoints={recording.live_trajectory || []}
            processedPoints={procSummary?.hand_trajectory_preview || []}
            isRecording={isRecording}
            latestWrist={camera.latest_wrist}
          />
        </section>

        {/* RIGHT PANEL — ROBOT SIMULATION INSIDE THE SAME WEBPAGE */}
        <section className="panel main-split-panel">
          <div className="panel-header">
            <div className="panel-title">
              RIGHT PANEL — EMBEDDED ROBOT SIMULATION (URDF + Live ROS 2 / Gazebo State)
            </div>
            <div className="panel-actions">
              <span
                className={`state-tag ${executionStateLabel.toLowerCase().replace(/\s+/g, '-')}`}
              >
                State: {executionStateLabel.toUpperCase()}
              </span>
              <button
                className="btn"
                onClick={handleConnectSim}
                disabled={!isBackendOnline || busyAction !== null}
              >
                Refresh ROS 2
              </button>
              {(!isGazeboRunning || !isControllerReady) && (
                <button
                  className="btn primary"
                  onClick={handleLaunchSim}
                  disabled={!isBackendOnline || busyAction !== null}
                >
                  Launch Gazebo Sim
                </button>
              )}
            </div>
          </div>

          {/* EMBEDDED 3D ROBOT VIEWPORT (Occupies majority of the right panel) */}
          <RobotViewport3D
            urdfSpec={urdfSpec}
            liveJointsRad={actualJointsRad}
            targetJointsRad={targetJointsRad}
            isGazeboRunning={isGazeboRunning}
            isControllerReady={isControllerReady}
            executionStatus={rawExecStatus}
            waypoints={waypoints}
            executedTrail={executedTrail}
            currentStep={robot.current_waypoint_step || 0}
            wsConnected={wsConnected}
          />

          {/* EXECUTION PROGRESS BAR */}
          <div className="execution-progress-box">
            <div className="execution-progress-header">
              <span>
                <strong>Trajectory Execution:</strong>{' '}
                {robot.active_demonstration_id || selectedDemoId || 'None'} (Step{' '}
                {robot.current_waypoint_step ?? 0} /{' '}
                {robot.total_waypoint_steps || waypoints.length || 0})
              </span>
              <span style={{ color: '#38bdf8', fontWeight: 700 }}>
                {robot.playback_progress_percent ?? 0}%
              </span>
            </div>
            <div className="progress-track">
              <div
                className="progress-fill"
                style={{ width: `${robot.playback_progress_percent ?? 0}%` }}
              />
            </div>
          </div>

          {/* ROBOT EXECUTION CONTROLS */}
          <div className="robot-controls-bar">
            <button
              className="btn success btn-lg"
              onClick={handleExecuteDemo}
              disabled={
                !isControllerReady ||
                !isDemoProcessed ||
                isRobotExecuting ||
                isRecording ||
                busyAction !== null
              }
              title={
                !isDemoProcessed
                  ? 'Record and validate a demonstration first'
                  : 'Execute the validated trajectory on the simulated robot'
              }
            >
              ▶ Execute Demonstration ({waypoints.length} pts)
            </button>

            <button
              className="btn"
              onClick={() => handlePauseResume(isPaused)}
              disabled={!isRobotExecuting || busyAction !== null}
              title="Pause or resume active trajectory execution"
            >
              {isPaused ? '▶ Resume' : '⏸ Pause'}
            </button>

            <button
              className="btn"
              onClick={handleResetRobot}
              disabled={!isControllerReady || busyAction !== null}
              title="Reset simulated robot arm to Home pose [0, 0, 0] rad"
            >
              ↺ Reset Sim (Home)
            </button>

            <button
              className="btn primary"
              onClick={handleTestMotion}
              disabled={!isControllerReady || isRobotExecuting || busyAction !== null}
              title="Run a 3-pose URDF-validated test motion"
            >
              Test Motion
            </button>

            <button
              className="btn stop-emergency"
              onClick={handleStopRobot}
              disabled={!isBackendOnline}
              title="Immediately stop active robot motion and hold current joint positions"
            >
              ■ STOP ROBOT
            </button>
          </div>
        </section>
      </main>

      {/* =================================================================== */}
      {/* LEFT-TO-RIGHT PROCESSING PIPELINE & COORDINATE CONVERSION BAR       */}
      {/* =================================================================== */}
      <section className="panel">
        <div className="panel-header">
          <div className="panel-title">
            Left-to-Right Processing Pipeline &amp; Human-to-Robot Coordinate Conversion
          </div>
          <div className="panel-actions">
            {procSummary && waypoints.length > 0 && (
              <button
                className="btn"
                onClick={() => setShowWaypointsTable((v) => !v)}
              >
                {showWaypointsTable
                  ? 'Hide Waypoint Table'
                  : `Show Validated Waypoints (${waypoints.length})`}
              </button>
            )}
            <button
              className="btn primary"
              onClick={() => handleProcessDemo(selectedDemoId)}
              disabled={!isBackendOnline || !selectedDemoId || isRecording || busyAction !== null}
            >
              Re-Validate &amp; Process Selected Recording
            </button>
          </div>
        </div>

        {/* 8-STAGE PIPELINE STRIP */}
        <div className="pipeline-eight-grid">
          {pipelineStages.map((stage) => (
            <div
              key={stage.id}
              className={`pipeline-stage-card ${stage.status || 'idle'}`}
            >
              <div className="stage-header-row">
                <strong>{stage.label}</strong>
                <span className="stage-dot" />
              </div>
              <span>{stage.detail}</span>
            </div>
          ))}
        </div>

        {/* EXPLICIT COORDINATE CONVERSION & BC MODEL COMPATIBILITY SUMMARY */}
        {procSummary && (
          <div className="conversion-summary-box">
            <div className="conversion-col">
              <div>
                <strong>Hand Reference &amp; Smoothing:</strong>{' '}
                {procSummary.reference_point || 'Wrist (Landmark 0) + Palm Centre (0,5,9,13,17)'} •{' '}
                {procSummary.coordinate_mapping_spec?.smoothing_filter}
              </div>
              <div>
                <strong>Coordinate Mapping:</strong> Camera ΔX → Robot Lateral Y{' '}
                <code>[-0.65, +0.65]m</code> (Joint 1 Yaw) | Camera -ΔY → Robot Vertical Z{' '}
                <code>[1.62, 2.26]m</code> (Joint 2/3 Pitch) | Depth/BC ΔX → Robot Forward X{' '}
                <code>[0.85, 1.25]m</code>
              </div>
            </div>
            <div className="conversion-col">
              <div>
                <strong>Mapping Mode:</strong>{' '}
                <span style={{ color: '#fbbf24' }}>
                  {procSummary.target_generation_label}
                </span>{' '}
                | <strong>FK Max Error:</strong>{' '}
                <span style={{ color: '#34d399' }}>
                  {procSummary.validation_report?.max_fk_error_m ?? 0} m
                </span>
              </div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)' }}>
                {procSummary.target_generation_explanation}
              </div>
            </div>
          </div>
        )}

        {/* COLLAPSIBLE WAYPOINT TABLE */}
        {showWaypointsTable && waypoints.length > 0 && (
          <div className="table-scroll">
            <table className="waypoints-table">
              <thead>
                <tr>
                  <th>Step</th>
                  <th>Frame</th>
                  <th>Time (s)</th>
                  <th>Hand Norm Δ [dx, dy, dz]</th>
                  <th>Raw BC Output [x, y, z] (m)</th>
                  <th>URDF-Safe Target [x, y, z] (m)</th>
                  <th>Validated Joints [j1, j2, j3] (rad)</th>
                  <th>FK Error (m)</th>
                </tr>
              </thead>
              <tbody>
                {waypoints.map((wp) => (
                  <tr key={wp.step}>
                    <td>{wp.step}</td>
                    <td>{wp.source_frame}</td>
                    <td>{wp.time_from_start_sec}s</td>
                    <td>[{wp.normalized_hand_xyz?.join(', ')}]</td>
                    <td>[{wp.raw_bc_xyz?.join(', ')}]</td>
                    <td>[{wp.target_xyz?.join(', ')}]</td>
                    <td style={{ color: '#34d399' }}>[{wp.joints_rad?.join(', ')}]</td>
                    <td>{wp.fk_error_m}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* =================================================================== */}
      {/* BOTTOM COMPACT STRIP: ROS STATUS | JOINT VALUES | SAVED DEMOS | LOGS*/}
      {/* =================================================================== */}
      <section className="bottom-diagnostics-grid">
        {/* CARD 1: ROS 2 & GAZEBO STATUS */}
        <div className="panel compact-card">
          <div className="panel-title">ROS 2 &amp; Gazebo Status</div>
          <div className="info-box">
            <div className="info-row">
              <span>ROS 2 Bridge:</span>
              <span style={{ color: isRos2Connected ? '#34d399' : '#f87171' }}>
                {isRos2Connected ? 'CONNECTED' : 'DISCONNECTED'}
              </span>
            </div>
            <div className="info-row">
              <span>Gazebo Sim 8:</span>
              <span style={{ color: isGazeboRunning ? '#34d399' : '#f87171' }}>
                {isGazeboRunning ? 'RUNNING' : 'OFFLINE'}
              </span>
            </div>
            <div className="info-row">
              <span>Trajectory Topic:</span>
              <span>/arm_controller/joint_trajectory ({robot.trajectory_subscribers ?? 0} sub)</span>
            </div>
            <div className="info-row">
              <span>/joint_states Stream:</span>
              <span>
                {robot.joint_state_msg_count ?? 0} msgs ({wsConnected ? 'WS 25Hz' : 'HTTP'})
              </span>
            </div>
          </div>
        </div>

        {/* CARD 2: LIVE JOINT VALUES */}
        <div className="panel compact-card">
          <div className="panel-title">Live Joint Values (`/joint_states` vs URDF Limits)</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.38rem' }}>
            {jointNames.map((jname, idx) => {
              const lim = jointLimits[jname] || { lower: -1.57, upper: 1.57 }
              const posRad = actualJointsRad ? actualJointsRad[idx] : null
              const posDeg = actualJointsDeg ? actualJointsDeg[idx] : null
              const pct = posRad !== null ? jointPercent(posRad, lim.lower, lim.upper) : 50
              return (
                <div className="joint-card" key={jname}>
                  <div className="joint-header">
                    <span>
                      <strong>{jname}</strong> [{lim.lower}, {lim.upper}]
                    </span>
                    <span style={{ color: posRad !== null ? '#34d399' : 'var(--text-muted)' }}>
                      {posRad !== null
                        ? `${posRad.toFixed(3)} rad (${posDeg?.toFixed(1)}°)`
                        : 'Offline'}
                    </span>
                  </div>
                  <div className="joint-bar-track">
                    <div
                      className="joint-bar-fill"
                      style={{ width: posRad !== null ? `${pct}%` : '0%' }}
                    />
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* CARD 3: SAVED RECORDINGS */}
        <div className="panel compact-card">
          <div className="panel-title">Saved Recordings ({demonstrations.length})</div>
          <select
            className="select-input"
            value={selectedDemoId}
            onChange={(e) => setSelectedDemoId(e.target.value)}
          >
            {demonstrations.length === 0 && (
              <option value="">No saved demonstrations</option>
            )}
            {demonstrations.map((d) => (
              <option key={d.demonstration_id} value={d.demonstration_id}>
                {d.demonstration_id} ({d.detected_hand_frames}/{d.frame_count} hand) [
                {d.processing_status}]
              </option>
            ))}
          </select>
          {selectedDemoDetail && (
            <div className="info-box">
              <div className="info-row">
                <span>Selected ID:</span>
                <span>{selectedDemoDetail.demonstration_id}</span>
              </div>
              <div className="info-row">
                <span>Hand Frames:</span>
                <span>
                  {selectedDemoDetail.detected_hand_frames} / {selectedDemoDetail.frame_count} (
                  {selectedDemoDetail.duration_seconds}s)
                </span>
              </div>
              <div className="info-row">
                <span>Validation:</span>
                <span
                  style={{
                    color:
                      selectedDemoDetail.processing_status === 'processed'
                        ? '#34d399'
                        : '#f87171',
                  }}
                >
                  {(selectedDemoDetail.processing_status || 'unprocessed').toUpperCase()}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* CARD 4: SYSTEM & TRAJECTORY LOGS */}
        <div className="panel compact-card">
          <div className="panel-title">
            <span>System &amp; Trajectory Logs</span>
            <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>
              {lastUpdated || ''}
            </span>
          </div>
          <div className="log-box compact-log-box">
            {(health?.robot?.recent_logs && health.robot.recent_logs.length > 0
              ? health.robot.recent_logs
              : ['[System] Ready. Click Start Recording, move your hand, then click Stop Recording.']
            ).map((entry, idx) => (
              <div key={idx}>{entry}</div>
            ))}
          </div>
        </div>
      </section>

      <footer className="dashboard-footer">
        <span>AI-Powered Robot Arm Learning from Human Demonstration — Integrated Two-Panel Console</span>
        <span>Dashboard: http://127.0.0.1:5173 | FastAPI + WebSocket: http://127.0.0.1:8000</span>
      </footer>
    </div>
  )
}
