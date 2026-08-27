import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from './api'
import type { Mix, MixOptions, Overview, PlaylistItem, Preparation, Track } from './types'

const emptyOverview: Overview = { preparations: [], mixes: [], activeJobs: 0 }

function minutes(seconds = 0) {
  const whole = Math.max(0, Math.floor(seconds))
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`
}

function statusLabel(status: string) {
  return status === 'ready' ? 'Ready' : status === 'running' ? 'Working' : status === 'failed' ? 'Needs attention' : 'Queued'
}

function hue(input: string) {
  return [...input].reduce((acc, letter) => (acc * 31 + letter.charCodeAt(0)) % 360, 18)
}

function Artwork({ track, spinning = false, size = 'normal' }: { track: Track; spinning?: boolean; size?: 'normal' | 'small' }) {
  const background = `linear-gradient(135deg, hsl(${hue(track.album)} 70% 55%), hsl(${(hue(track.artist) + 55) % 360} 75% 23%))`
  return (
    <div className={`artwork ${size} ${spinning ? 'spinning' : ''}`} style={{ background }}>
      {track.artworkUrl ? <img src={track.artworkUrl} alt={`${track.album} artwork`} /> : <span>{track.title.slice(0, 1)}</span>}
      <i />
    </div>
  )
}

const fallbackWaveform = [0.18, 0.4, 0.68, 0.36, 0.88, 0.54, 0.74, 0.32, 0.62, 0.92, 0.47, 0.23, 0.76, 0.55, 0.94, 0.37, 0.64, 0.82, 0.44, 0.24, 0.68, 0.42, 0.86, 0.57, 0.31, 0.73, 0.52, 0.91, 0.39, 0.63, 0.28, 0.77, 0.48, 0.89, 0.35, 0.7, 0.5, 0.97, 0.61, 0.29]

function clamp(value: number, lower: number, upper: number) {
  return Math.min(upper, Math.max(lower, value))
}

function sampleWaveform(points: number[], position: number) {
  const normalised = clamp(position, 0, 1) * (points.length - 1)
  const left = Math.floor(normalised)
  const right = Math.min(points.length - 1, left + 1)
  const fraction = normalised - left
  return points[left] + (points[right] - points[left]) * fraction
}

function resampleWaveform(points: number[], count: number) {
  return Array.from({ length: count }, (_, index) => sampleWaveform(points, index / Math.max(1, count - 1)))
}

function waveformPath(points: number[], center: number, amplitude: number, width = 1000) {
  const top = points.map((value, index) => {
    const x = index / Math.max(1, points.length - 1) * width
    return `${index ? 'L' : 'M'}${x.toFixed(2)},${(center - value * amplitude).toFixed(2)}`
  })
  const bottom = [...points].reverse().map((value, reverseIndex) => {
    const index = points.length - 1 - reverseIndex
    const x = index / Math.max(1, points.length - 1) * width
    return `L${x.toFixed(2)},${(center + value * amplitude).toFixed(2)}`
  })
  return [...top, ...bottom, 'Z'].join(' ')
}

function fallbackBeats(track: Track) {
  const beatLength = 60 / Math.max(60, track.bpm || 120)
  return Array.from({ length: Math.max(1, Math.ceil(track.durationSeconds / beatLength)) }, (_, index) => Number((index * beatLength).toFixed(3)))
}

function libraryUrl(relativePath: string) {
  return `/library/${relativePath.split('/').map(encodeURIComponent).join('/')}`
}

function decodedWaveform(buffer: AudioBuffer, count = 2048) {
  const channel = buffer.getChannelData(0)
  const points = Array.from({ length: count }, (_, index) => {
    const start = Math.floor(index / count * channel.length)
    const end = Math.max(start + 1, Math.floor((index + 1) / count * channel.length))
    let peak = 0
    for (let cursor = start; cursor < end; cursor += 1) peak = Math.max(peak, Math.abs(channel[cursor]))
    return peak
  })
  const peak = Math.max(...points, 0)
  return peak ? points.map(point => point / peak) : points
}

function waveformDetailPointCount(track: Track) {
  return Math.max(512, Math.ceil(track.durationSeconds * Math.max(track.bpm, 60) / 60 * 16))
}

function decodeWaveformDetail(encoded?: string | null) {
  if (!encoded) return null
  try {
    const bytes = window.atob(encoded)
    return Array.from(bytes, value => value.charCodeAt(0) / 255)
  } catch {
    return null
  }
}

function Waveform({ active = true, track, item, elapsed }: { active?: boolean; track: Track; item: PlaylistItem; elapsed: number }) {
  const [decoded, setDecoded] = useState<{ trackId: string; points: number[] } | null>(null)
  const storedDetail = useMemo(() => decodeWaveformDetail(track.waveformDetail), [track.waveformDetail])
  useEffect(() => {
    if (storedDetail) return
    let disposed = false
    const context = new AudioContext()
    void fetch(libraryUrl(track.relativePath))
      .then(response => response.ok ? response.arrayBuffer() : Promise.reject(new Error('Could not read local audio')))
      .then(data => context.decodeAudioData(data))
      .then(audioBuffer => { if (!disposed) setDecoded({ trackId: track.id, points: decodedWaveform(audioBuffer, waveformDetailPointCount(track)) }) })
      .catch(() => undefined)
      .finally(() => { void context.close() })
    return () => { disposed = true; void context.close() }
  }, [storedDetail, track.id, track.relativePath, track.durationSeconds, track.bpm])
  const points = useMemo(() => {
    const source = storedDetail ?? (decoded?.trackId === track.id ? decoded.points : track.waveform)
    return source.length ? source.map(value => clamp(value, 0, 1)) : fallbackWaveform
  }, [decoded, track.id, track.waveform])
  const overviewPoints = useMemo(() => resampleWaveform(points, Math.min(512, Math.max(128, points.length))), [points])
  const duration = Math.max(1, track.durationSeconds)
  const sourceStart = clamp(item.sourceStartSeconds ?? 0, 0, duration)
  const sourceEnd = clamp(item.sourceEndSeconds ?? duration, sourceStart, duration) || duration
  const tempoRatio = item.deckBpm && track.bpm ? item.deckBpm / track.bpm : 1
  const trackElapsed = active ? Math.max(0, elapsed - item.startSeconds) : 0
  const sourcePosition = clamp(sourceStart + trackElapsed * tempoRatio, sourceStart, sourceEnd)
  const windowSeconds = Math.min(10, duration)
  // Put the playhead one eighth from the left: the deck is forward-looking,
  // while retaining just enough past waveform to judge the beat alignment.
  const windowStart = clamp(sourcePosition - windowSeconds * 0.125, 0, Math.max(0, duration - windowSeconds))
  const beats = track.beatGrid.length ? track.beatGrid : fallbackBeats(track)
  const downbeats = new Set((track.downbeats.length ? track.downbeats : beats.filter((_, index) => index % 4 === 0)).map(beat => beat.toFixed(3)))
  const visibleBeats = beats.filter(beat => beat >= windowStart && beat <= windowStart + windowSeconds)
  const topPointCount = Math.max(256, visibleBeats.length * 16)
  const zoomedPoints = Array.from({ length: topPointCount }, (_, index) => sampleWaveform(points, (windowStart + index / Math.max(1, topPointCount - 1) * windowSeconds) / duration))
  const viewportX = windowStart / duration * 1000
  const viewportWidth = Math.max(5, windowSeconds / duration * 1000)
  const playheadX = (sourcePosition - windowStart) / windowSeconds * 1000
  return <div className={`track-waveform ${active ? 'active' : ''}`} aria-label="Live beat waveform and full-track energy overview">
    <svg viewBox="0 0 1000 234" preserveAspectRatio="none" role="img">
      <rect className="wave-bg" x="0" y="0" width="1000" height="108" rx="8" />
      {visibleBeats.map(beat => {
        const x = (beat - windowStart) / windowSeconds * 1000
        const downbeat = downbeats.has(beat.toFixed(3))
        return <line key={`live-${beat}`} className={downbeat ? 'downbeat-line' : 'beat-line'} x1={x} x2={x} y1={downbeat ? 8 : 80} y2="100" />
      })}
      <path className="zoom-wave-fill" d={waveformPath(zoomedPoints, 54, 38)} />
      <line className="playhead-line" x1={playheadX} x2={playheadX} y1="4" y2="104" />
      <text className="wave-label" x="12" y="20">LIVE · BEAT GRID</text>
      <rect className="overview-bg" x="0" y="126" width="1000" height="100" rx="8" />
      {beats.filter(beat => downbeats.has(beat.toFixed(3))).map(beat => {
        const x = beat / duration * 1000
        return <line key={`overview-${beat}`} className="overview-downbeat" x1={x} x2={x} y1="132" y2="220" />
      })}
      <path className="overview-energy-fill" d={waveformPath(overviewPoints, 180, 33)} />
      <rect className="viewport-window" x={viewportX} y="129" width={viewportWidth} height="94" rx="4" />
      <text className="wave-label" x="12" y="146">FULL TRACK · ENERGY</text>
    </svg>
  </div>
}

function Icon({ name, size = 18 }: { name: 'play' | 'pause' | 'skip' | 'volume' | 'plus' | 'library' | 'spark' | 'queue' | 'grip' | 'trash'; size?: number }) {
  const paths = {
    play: <path d="M6 4.5v15l12-7.5-12-7.5Z" fill="currentColor" />,
    pause: <><path d="M6 4h4v16H6z" fill="currentColor" /><path d="M14 4h4v16h-4z" fill="currentColor" /></>,
    skip: <><path d="m5 5 9 7-9 7V5Z" fill="currentColor" /><path d="m14 5 6 7-6 7V5Z" fill="currentColor" /><path d="M20 5h2v14h-2z" fill="currentColor" /></>,
    volume: <><path d="M4 9h4l5-4v14l-5-4H4V9Z" fill="none" stroke="currentColor" strokeWidth="2" /><path d="M16 9a4 4 0 0 1 0 6M19 6a8 8 0 0 1 0 12" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" /></>,
    plus: <path d="M12 5v14M5 12h14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />,
    library: <><path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v15.5A2.5 2.5 0 0 0 17.5 16H6.5A2.5 2.5 0 0 0 4 18.5v-13Z" fill="none" stroke="currentColor" strokeWidth="1.8" /><path d="M8 7h8M8 11h8" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" /></>,
    spark: <path d="m12 2 1.8 6.2L20 10l-6.2 1.8L12 18l-1.8-6.2L4 10l6.2-1.8L12 2Z" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />,
    queue: <><path d="M5 6h14M5 12h14M5 18h9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" /><path d="m17 16 3 2-3 2" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></>,
    grip: <path d="M9 6h.01M15 6h.01M9 12h.01M15 12h.01M9 18h.01M15 18h.01" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />,
    trash: <><path d="M4 7h16M9 7V4h6v3M7 7l1 13h8l1-13M10 11v5M14 11v5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></>,
  }
  return <svg viewBox="0 0 24 24" width={size} height={size} aria-hidden="true">{paths[name]}</svg>
}

function JobPill({ status }: { status: string }) {
  return <span className={`status ${status}`}><i />{statusLabel(status)}</span>
}

function preparationSummary(preparation: Preparation) {
  if (preparation.status === 'ready') return `${preparation.trackCount} tracks`
  if (preparation.discoveredTrackCount) {
    const processed = preparation.analysedTrackCount + preparation.failedTrackCount
    return `${processed}/${preparation.discoveredTrackCount} tracks analysed`
  }
  return preparation.message
}

function Sidebar({ overview, selectedPrep, selectedMix, deletingId, onSelectPrep, onSelectMix, onDeletePreparation, onDeleteMix, onNew }: {
  overview: Overview; selectedPrep?: string; selectedMix?: string; deletingId?: string; onSelectPrep: (id: string) => void; onSelectMix: (id: string) => void; onDeletePreparation: (preparation: Preparation) => void; onDeleteMix: (mix: Mix) => void; onNew: () => void
}) {
  return <aside className="sidebar">
    <div className="brand"><div className="brand-mark">A</div><span>Attatouille</span></div>
    <button className="new-button" onClick={onNew}><Icon name="plus" />Prepare a folder</button>
    <nav>
      <p className="nav-heading"><Icon name="library" size={15} /> Libraries</p>
      {overview.preparations.length === 0 && <span className="empty-nav">No prepared folders yet</span>}
      {overview.preparations.map(prep => (
        <div className="nav-row" key={prep.id}><button className={`nav-item ${selectedPrep === prep.id ? 'selected' : ''}`} onClick={() => onSelectPrep(prep.id)}>
          <span className="nav-copy"><b>{prep.label}</b><small>{preparationSummary(prep)}</small></span><JobPill status={prep.status} />
        </button><button className="delete-item" disabled={deletingId === prep.id} onClick={() => onDeletePreparation(prep)} aria-label={`${['queued', 'running'].includes(prep.status) ? 'Cancel and delete' : 'Delete'} ${prep.label}`} title={['queued', 'running'].includes(prep.status) ? 'Cancel and delete preparation' : 'Delete preparation'}><Icon name="trash" size={15} /></button></div>
      ))}
      <p className="nav-heading mixes"><Icon name="queue" size={15} /> Mixes</p>
      {overview.mixes.length === 0 && <span className="empty-nav">Your prepared mixes appear here</span>}
      {overview.mixes.map(mix => (
        <div className="nav-row" key={mix.id}><button className={`nav-item ${selectedMix === mix.id ? 'selected' : ''}`} onClick={() => onSelectMix(mix.id)}>
          <span className="nav-copy"><b>{mix.name}</b><small>{mix.playlist.length ? `${mix.playlist.length} tracks · ${minutes(mix.durationSeconds)}` : mix.message}</small></span><JobPill status={mix.status} />
        </button><button className="delete-item" disabled={deletingId === mix.id} onClick={() => onDeleteMix(mix)} aria-label={`${['queued', 'running'].includes(mix.status) ? 'Cancel and delete' : 'Delete'} ${mix.name}`} title={['queued', 'running'].includes(mix.status) ? 'Cancel and delete mix' : 'Delete mix'}><Icon name="trash" size={15} /></button></div>
      ))}
    </nav>
    <div className="local-note"><span>Local-first</span><p>No uploads. Models and music stay on this device.</p></div>
  </aside>
}

function FolderPreparation({ onCreated, busy }: { onCreated: (prep: Preparation) => void; busy: boolean }) {
  const [label, setLabel] = useState('Party crate')
  const [path, setPath] = useState('/music')
  const [error, setError] = useState('')
  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    setError('')
    try { onCreated(await api.createPreparation({ label, sourcePath: path })) } catch (problem) { setError(problem instanceof Error ? problem.message : 'Could not start preparation') }
  }
  return <section className="setup-card">
    <div className="eyebrow"><Icon name="spark" size={15} /> Step 1 · Library preparation</div>
    <h1>Turn a music folder into a <em>mixable crate.</em></h1>
    <p className="lede">We read tags, beat grids, keys, song structure, energy and local 512-dimensional music embeddings before planning any transitions.</p>
    <form onSubmit={submit}>
      <label>Crate name<input value={label} onChange={event => setLabel(event.target.value)} maxLength={60} required /></label>
      <label>Folder inside the mounted library<input value={path} onChange={event => setPath(event.target.value)} placeholder="/music/Friday" required /></label>
      <p className="helper">Compose maps your computer’s chosen folder to <code>/music</code>. Use a subfolder such as <code>/music/Friday</code> when needed.</p>
      {error && <p className="form-error">{error}</p>}
      <button className="primary-button" disabled={busy}><Icon name="spark" />{busy ? 'Preparation running…' : 'Prepare this folder'}</button>
    </form>
    <div className="feature-row"><span>Harmonix structure</span><span>Local genre inference</span><span>Qdrant vectors</span></div>
  </section>
}

function ProgressCard({ item }: { item: Preparation | Mix }) {
  return <section className="progress-card">
    <div className="progress-top"><div><div className="eyebrow">Background task</div><h2>{item instanceof Object && 'label' in item ? item.label : item.name}</h2></div><JobPill status={item.status} /></div>
    <p>{item.message}</p>
    <div className="progress-line"><i style={{ width: `${item.progress}%` }} /></div>
    <small>{item.progress}% complete · You can prepare another folder while this runs.</small>
  </section>
}

const analysisWaveBars = [30, 55, 82, 44, 70, 94, 58, 36, 76, 49, 88, 61, 33, 67, 97, 53, 79, 41, 91, 64, 35, 72, 48, 86, 57, 39, 74, 96, 52, 68, 43, 83]

function AnalysisWaveLoader() {
  return <div className="analysis-wave-loader" aria-label="Analysing audio waveform" role="img">
    <div className="analysis-wave-baseline" />
    {analysisWaveBars.map((height, index) => <i key={index} style={{ '--wave-height': `${height}%`, '--wave-delay': `${-index * 0.07}s` } as React.CSSProperties} />)}
  </div>
}

function PreparationProgressCard({ preparation }: { preparation: Preparation }) {
  const total = preparation.discoveredTrackCount
  const analysed = preparation.analysedTrackCount
  const failed = preparation.failedTrackCount
  const processed = analysed + failed
  const currentPosition = preparation.currentTrack ? Math.min(total, processed + 1) : processed
  const active = ['queued', 'running'].includes(preparation.status)
  return <section className="progress-card preparation-progress" aria-live="polite">
    <div className="progress-top"><div><div className="eyebrow">Library preparation</div><h2>{preparation.label}</h2></div><JobPill status={preparation.status} /></div>
    {active && <AnalysisWaveLoader />}
    <p>{preparation.message}</p>
    {total > 0 && <div className="analysis-details">
      <div><span>Tracks found</span><b>{total}</b></div>
      <div><span>Progress</span><b>{currentPosition} / {total}</b></div>
      {failed > 0 && <div><span>Skipped</span><b>{failed}</b></div>}
    </div>}
    {preparation.currentTrack && <div className="current-analysis"><span>Analysing now</span><b title={preparation.currentTrack}>{preparation.currentTrack}</b></div>}
    <div className="progress-line"><i style={{ width: `${preparation.progress}%` }} /></div>
    <small>{preparation.progress}% complete{total > 0 ? ` · ${analysed} successfully analysed` : ' · scanning your music folder'}</small>
  </section>
}

function GenreOrder({ genres, onChange }: { genres: string[]; onChange: (genres: string[]) => void }) {
  const [draggedGenre, setDraggedGenre] = useState<string | null>(null)
  const [dropTarget, setDropTarget] = useState<string | null>(null)
  const commitMove = (targetGenre: string) => {
    if (!draggedGenre || draggedGenre === targetGenre) return
    const source = genres.indexOf(draggedGenre)
    const target = genres.indexOf(targetGenre)
    if (source < 0 || target < 0) return
    const reordered = [...genres]
    reordered.splice(source, 1)
    reordered.splice(target, 0, draggedGenre)
    onChange(reordered)
  }
  return <div className="genre-order">
    <div className="field-heading"><span>Genre journey</span><small>Drag to set the flow</small></div>
    <div className="genre-list">
      {genres.map((genre, index) => <div className={`genre-chip ${draggedGenre === genre ? 'dragging' : ''} ${dropTarget === genre ? 'drop-target' : ''}`} key={genre} draggable
        onDragStart={event => { event.dataTransfer.effectAllowed = 'move'; event.dataTransfer.setData('text/plain', genre); setDraggedGenre(genre) }}
        onDragOver={event => { event.preventDefault(); event.dataTransfer.dropEffect = 'move'; setDropTarget(genre) }}
        onDrop={event => { event.preventDefault(); commitMove(genre); setDraggedGenre(null); setDropTarget(null) }}
        onDragLeave={() => setDropTarget(current => current === genre ? null : current)}
        onDragEnd={() => { setDraggedGenre(null); setDropTarget(null) }}>
        <Icon name="grip" size={15} /><span>{index + 1}</span>{genre}
      </div>)}
    </div>
  </div>
}

function MixSetup({ preparation, onCreated }: { preparation: Preparation; onCreated: (mix: Mix) => void }) {
  const [genres, setGenres] = useState(preparation.genres)
  const preparationId = useRef(preparation.id)
  const [minSeconds, setMinSeconds] = useState(90)
  const [maxSeconds, setMaxSeconds] = useState(180)
  const [acceptance, setAcceptance] = useState(85)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  // The overview is polled every few seconds, creating a new `genres` array
  // each time. Reset only when the user actually selects another preparation;
  // otherwise a server refresh would erase a drag-and-drop reorder.
  useEffect(() => {
    if (preparationId.current !== preparation.id) {
      preparationId.current = preparation.id
      setGenres(preparation.genres)
    }
  }, [preparation.id, preparation.genres])
  const createMix = async () => {
    setLoading(true); setError('')
    try {
      const options: MixOptions = { preparationId: preparation.id, genreOrder: genres, minTrackSeconds: minSeconds, maxTrackSeconds: maxSeconds, acceptancePercentage: acceptance }
      onCreated(await api.createMix(options))
    } catch (problem) { setError(problem instanceof Error ? problem.message : 'Could not create the mix') }
    finally { setLoading(false) }
  }
  return <section className="mix-setup">
    <div className="eyebrow"><Icon name="spark" size={15} /> Step 2 · Mix design</div>
    <div className="setup-title"><div><h1>Build a mix from <em>{preparation.label}.</em></h1><p>{preparation.trackCount} analysed tracks · all transition points are phrase-safe and quality scored.</p></div><span className="ready-stamp">Data preparation done</span></div>
    <GenreOrder genres={genres} onChange={setGenres} />
    <div className="controls-grid">
      <label className="range-field"><span>Minimum per track <b>{Math.floor(minSeconds / 60)}m {minSeconds % 60}s</b></span><input type="range" min="30" max="300" step="15" value={minSeconds} onChange={event => setMinSeconds(Math.min(Number(event.target.value), maxSeconds - 15))} /></label>
      <label className="range-field"><span>Maximum per track <b>{Math.floor(maxSeconds / 60)}m {maxSeconds % 60}s</b></span><input type="range" min="45" max="360" step="15" value={maxSeconds} onChange={event => setMaxSeconds(Math.max(Number(event.target.value), minSeconds + 15))} /></label>
      <label className="range-field acceptance"><span>Track acceptance <b>{acceptance}%</b></span><input type="range" min="50" max="100" step="5" value={acceptance} onChange={event => setAcceptance(Number(event.target.value))} /><small>At most {Math.floor(preparation.trackCount * (100 - acceptance) / 100)} poor-fit tracks can be left out.</small></label>
    </div>
    {error && <p className="form-error">{error}</p>}
    <button className="primary-button create-mix" onClick={createMix} disabled={loading || !genres.length}><Icon name="spark" />{loading ? 'Queueing mix…' : 'Create mix from preparation'}</button>
    {preparation.modelReport && <p className="model-note">{preparation.modelReport.structureModel} · {preparation.modelReport.embeddingModel}</p>}
  </section>
}

function Deck({ item, role, playing, elapsed }: { item?: PlaylistItem; role: 'now' | 'next'; playing: boolean; elapsed: number }) {
  if (!item) return <div className="deck empty-deck"><span>{role === 'next' ? 'No next track' : 'Select a mix'}</span></div>
  const { track } = item
  const gain = item.gainDb ?? 0
  const gainLabel = `Auto gain ${gain >= 0 ? '+' : ''}${gain.toFixed(1)} dB`
  return <article className={`deck ${role}`}>
    <div className="deck-label"><span>{role === 'now' ? 'On air' : 'Up next'}</span><span>{role === 'now' ? `${track.bpm.toFixed(0)} BPM · ${track.key} · ${gainLabel}` : `in ${minutes(Math.max(0, item.startSeconds - elapsed))} · ${gainLabel}`}</span></div>
    <Artwork track={track} spinning={role === 'now' && playing} />
    <div className="track-copy"><h2>{track.title}</h2><p>{track.artist} <i>·</i> {track.album}</p></div>
    <Waveform active={role === 'now' && playing} track={track} item={item} elapsed={elapsed} />
    {item.transitionIn && <div className="transition-label"><span>{item.transitionIn.style}</span><b>{(item.transitionIn.renderQualityScore ?? item.transitionIn.qualityScore).toFixed(0)}% fit</b></div>}
  </article>
}

function Player({ mix }: { mix: Mix }) {
  const audio = useRef<HTMLAudioElement>(null)
  const bridgeAudio = useRef<HTMLAudioElement>(null)
  const [playing, setPlaying] = useState(false)
  const [bridging, setBridging] = useState(false)
  const [skipping, setSkipping] = useState(false)
  const [volume, setVolume] = useState(0.8)
  const [elapsed, setElapsed] = useState(0)
  const [notice, setNotice] = useState('')
  const timer = useRef<number | null>(null)
  const animationFrame = useRef<number | null>(null)
  const fadeTo = useCallback((player: HTMLAudioElement | null, target: number, duration: number) => new Promise<void>(resolve => {
    if (!player) { resolve(); return }
    if (timer.current) window.clearInterval(timer.current)
    const origin = player.volume
    const started = performance.now()
    timer.current = window.setInterval(() => {
      const progress = Math.min(1, (performance.now() - started) / duration)
      player.volume = origin + (target - origin) * progress
      if (progress === 1) { if (timer.current) window.clearInterval(timer.current); resolve() }
    }, 25)
  }), [])
  useEffect(() => () => { if (timer.current) window.clearInterval(timer.current); if (animationFrame.current) window.cancelAnimationFrame(animationFrame.current); bridgeAudio.current?.pause() }, [])
  useEffect(() => {
    if (!playing || bridging) return
    const paintFrame = () => {
      if (audio.current) {
        const nextElapsed = audio.current.currentTime
        setElapsed(current => Math.abs(current - nextElapsed) > 0.004 ? nextElapsed : current)
      }
      animationFrame.current = window.requestAnimationFrame(paintFrame)
    }
    animationFrame.current = window.requestAnimationFrame(paintFrame)
    return () => { if (animationFrame.current) window.cancelAnimationFrame(animationFrame.current) }
  }, [playing, bridging])
  const currentIndex = Math.max(0, mix.playlist.findIndex((item, index) => elapsed >= item.startSeconds && (index === mix.playlist.length - 1 || elapsed < mix.playlist[index + 1].startSeconds)))
  const current = mix.playlist[currentIndex] ?? mix.playlist[0]
  const next = mix.playlist[currentIndex + 1]
  const toggle = async () => {
    const player = bridging ? bridgeAudio.current : audio.current
    if (!player || !mix.audioUrl) return
    if (playing) { await fadeTo(player, 0, 2000); player.pause(); setPlaying(false) }
    else { player.volume = 0; await player.play(); setPlaying(true); await fadeTo(player, volume, 2000) }
  }
  const skip = async () => {
    if (!next || skipping) return
    setSkipping(true)
    try {
      const plan = await api.nextTransition(mix.id, currentIndex, elapsed)
      const transitionQuality = plan.transition.renderQualityScore ?? plan.transition.qualityScore
      setNotice(`Playing ${plan.transition.style} · ${transitionQuality.toFixed(0)}% transition fit`)
      const player = audio.current
      const bridge = bridgeAudio.current
      if (player && bridge) {
        const wasPlaying = playing
        player.pause()
        bridge.src = plan.audioUrl
        bridge.currentTime = 0
        bridge.volume = volume
        bridge.onended = async () => {
          setBridging(false)
          player.currentTime = plan.resumeAtSeconds
          setElapsed(plan.resumeAtSeconds)
          if (wasPlaying) {
            try { await player.play() } catch { setPlaying(false) }
          }
        }
        if (wasPlaying) {
          setBridging(true)
          await bridge.play()
        } else {
          player.currentTime = plan.resumeAtSeconds
          setElapsed(plan.resumeAtSeconds)
        }
      }
    } catch (problem) { setNotice(problem instanceof Error ? problem.message : 'Could not find a safe skip') }
    finally { setSkipping(false) }
  }
  const setPlayerVolume = (nextVolume: number) => {
    setVolume(nextVolume)
    if (audio.current) audio.current.volume = nextVolume
    if (bridgeAudio.current) bridgeAudio.current.volume = nextVolume
  }
  return <section className="player">
    <audio ref={audio} src={mix.audioUrl ?? undefined} onTimeUpdate={event => setElapsed(event.currentTarget.currentTime)} onEnded={() => setPlaying(false)} />
    <audio ref={bridgeAudio} onEnded={() => setBridging(false)} />
    <div className="player-top"><div><div className="eyebrow"><Icon name="queue" size={15} /> Prepared mix</div><h1>{mix.name}</h1></div><div className="mix-meta"><span>{mix.playlist.length} tracks</span><span>{minutes(mix.durationSeconds)}</span>{!mix.audioUrl && <b>Audio render unavailable</b>}</div></div>
    <div className="decks"><Deck item={current} role="now" playing={playing} elapsed={elapsed} /><Deck item={next} role="next" playing={playing} elapsed={elapsed} /></div>
    <div className="transport"><div className="time"><span>{minutes(elapsed)}</span><div><i style={{ width: `${mix.durationSeconds ? Math.min(100, elapsed / mix.durationSeconds * 100) : 0}%` }} /></div><span>{minutes(mix.durationSeconds)}</span></div><div className="player-actions"><button className="volume-button" aria-label="Volume"><Icon name="volume" /><input aria-label="Volume level" type="range" min="0" max="1" step="0.01" value={volume} onChange={event => setPlayerVolume(Number(event.target.value))} /></button><button className="play-button" onClick={toggle} disabled={!mix.audioUrl} aria-label={playing ? 'Pause mix' : 'Play mix'}><Icon name={playing ? 'pause' : 'play'} size={24} /></button><button className="skip-button" onClick={skip} disabled={!next || !mix.audioUrl || skipping} aria-label="Find a natural next transition"><Icon name="skip" />{skipping ? 'Finding…' : 'Next'}</button></div>{notice && <p className="player-notice">{notice}</p>}</div>
    <div className="mobile-transport"><button className="play-button" onClick={toggle} disabled={!mix.audioUrl}><Icon name={playing ? 'pause' : 'play'} size={24} /></button><button className="skip-button" onClick={skip} disabled={!next || !mix.audioUrl || skipping}><Icon name="skip" />{skipping ? 'Finding…' : 'Next'}</button><label><Icon name="volume" /><input aria-label="Volume level" type="range" min="0" max="1" step="0.01" value={volume} onChange={event => setPlayerVolume(Number(event.target.value))} /></label></div>
  </section>
}

function LibraryReady({ preparation }: { preparation: Preparation }) {
  return <section className="library-ready"><div className="eyebrow"><Icon name="library" size={15} /> Analysed crate</div><h1>{preparation.label}</h1><p>{preparation.trackCount} tracks are ready. Choose a genre order to turn it into a continuous mix.</p><div className="track-table"><div className="track-row header"><span>Track</span><span>Genre</span><span>BPM</span><span>Key</span><span>Energy</span></div>{preparation.tracks.slice(0, 7).map(track => <div className="track-row" key={track.id}><span><Artwork track={track} size="small" /><b>{track.title}<small>{track.artist}</small></b></span><span>{track.genres.join(', ')}</span><span>{track.bpm.toFixed(0)}</span><span>{track.key}</span><span><i className="energy-bar"><b style={{ width: `${track.energy * 100}%` }} /></i></span></div>)}</div></section>
}

export default function App() {
  const [overview, setOverview] = useState<Overview>(emptyOverview)
  const [selectedPreparationId, setSelectedPreparationId] = useState<string>()
  const [selectedMixId, setSelectedMixId] = useState<string>()
  const [showCreate, setShowCreate] = useState(true)
  const [error, setError] = useState('')
  const [deletingId, setDeletingId] = useState<string>()
  const load = useCallback(async () => {
    try { setOverview(await api.overview()); setError('') } catch (problem) { setError(problem instanceof Error ? problem.message : 'The local API is unavailable') }
  }, [])
  useEffect(() => {
    void load()
    const interval = window.setInterval(load, overview.activeJobs > 0 ? 1000 : 5000)
    return () => window.clearInterval(interval)
  }, [load, overview.activeJobs])
  const selectedPreparation = useMemo(() => overview.preparations.find(item => item.id === selectedPreparationId), [overview.preparations, selectedPreparationId])
  const selectedMix = useMemo(() => overview.mixes.find(item => item.id === selectedMixId), [overview.mixes, selectedMixId])
  const choosePreparation = (id: string) => { setSelectedPreparationId(id); setSelectedMixId(undefined); setShowCreate(false) }
  const chooseMix = (id: string) => { setSelectedMixId(id); setShowCreate(false) }
  const createdPreparation = (item: Preparation) => { setOverview(current => ({ ...current, preparations: [item, ...current.preparations] })); choosePreparation(item.id) }
  const createdMix = (item: Mix) => { setOverview(current => ({ ...current, mixes: [item, ...current.mixes] })); chooseMix(item.id) }
  const deletePreparation = async (preparation: Preparation) => {
    const mixCount = overview.mixes.filter(mix => mix.options.preparationId === preparation.id).length
    const related = mixCount ? ` and its ${mixCount} prepared mix${mixCount === 1 ? '' : 'es'}` : ''
    const cancelling = ['queued', 'running'].includes(preparation.status)
    if (!window.confirm(`${cancelling ? 'Cancel and delete' : 'Delete'} “${preparation.label}”${related}? This removes only analysis data and generated audio, never the source music folder.`)) return
    setDeletingId(preparation.id); setError('')
    try {
      const result = await api.deletePreparation(preparation.id)
      setOverview(current => ({
        ...current,
        preparations: current.preparations.filter(item => item.id !== result.deletedPreparationId),
        mixes: current.mixes.filter(item => !result.deletedMixIds.includes(item.id)),
      }))
      if (selectedPreparationId === preparation.id) setSelectedPreparationId(undefined)
      if (selectedMixId && result.deletedMixIds.includes(selectedMixId)) setSelectedMixId(undefined)
      setShowCreate(true)
    } catch (problem) { setError(problem instanceof Error ? problem.message : 'Could not delete the preparation') }
    finally { setDeletingId(undefined) }
  }
  const deleteMix = async (mix: Mix) => {
    const cancelling = ['queued', 'running'].includes(mix.status)
    if (!window.confirm(`${cancelling ? 'Cancel and delete' : 'Delete'} “${mix.name}”? This removes the generated mix audio, but keeps the prepared library and source music.`)) return
    setDeletingId(mix.id); setError('')
    try {
      await api.deleteMix(mix.id)
      setOverview(current => ({ ...current, mixes: current.mixes.filter(item => item.id !== mix.id) }))
      if (selectedMixId === mix.id) { setSelectedMixId(undefined); setShowCreate(true) }
    } catch (problem) { setError(problem instanceof Error ? problem.message : 'Could not delete the mix') }
    finally { setDeletingId(undefined) }
  }
  const working = Boolean(selectedPreparation && ['queued', 'running'].includes(selectedPreparation.status)) || Boolean(selectedMix && ['queued', 'running'].includes(selectedMix.status))
  return <div className="app-shell"><Sidebar overview={overview} selectedPrep={selectedPreparationId} selectedMix={selectedMixId} deletingId={deletingId} onSelectPrep={choosePreparation} onSelectMix={chooseMix} onDeletePreparation={deletePreparation} onDeleteMix={deleteMix} onNew={() => { setShowCreate(true); setSelectedPreparationId(undefined); setSelectedMixId(undefined) }} />
    <main><header><div className="live-dot"><i />Local session</div><div className="header-right">{overview.activeJobs > 0 && <span className="job-count">{overview.activeJobs} task{overview.activeJobs > 1 ? 's' : ''} running</span>}<span>DJ.Attatouille v0.1</span></div></header>
      {error && <div className="api-error">{error}. Start the stack with <code>docker compose up --build</code>.</div>}
      <div className="content">{showCreate ? <FolderPreparation busy={overview.activeJobs > 0} onCreated={createdPreparation} /> : selectedMix ? selectedMix.status === 'ready' ? <Player mix={selectedMix} /> : <ProgressCard item={selectedMix} /> : selectedPreparation ? selectedPreparation.status === 'ready' ? <><MixSetup preparation={selectedPreparation} onCreated={createdMix} /><LibraryReady preparation={selectedPreparation} /></> : <PreparationProgressCard preparation={selectedPreparation} /> : <FolderPreparation busy={working} onCreated={createdPreparation} />}</div>
    </main></div>
}
