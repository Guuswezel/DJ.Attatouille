export type JobStatus = 'queued' | 'running' | 'ready' | 'failed'

export interface Segment {
  start: number
  end: number
  label: string
  energy: number
}

export interface PhraseState {
  start: number
  end: number
  phraseIndex: number
  energy: number
  energySlope: number
  bassActivity: number
  drumActivity: number
  vocalActivity: number
  spectralDensity: number
  harmonicDensity: number
  noveltyIn: number
  noveltyOut: number
  loopability: number
  cueConfidence: number
}

export interface Track {
  id: string
  relativePath: string
  title: string
  artist: string
  album: string
  artworkUrl?: string | null
  durationSeconds: number
  bpm: number
  key: string
  energy: number
  loudnessLufs?: number | null
  waveform: number[]
  waveformDetail?: string | null
  beatGrid: number[]
  downbeats: number[]
  genres: string[]
  segments: Segment[]
  phraseStates: PhraseState[]
  cues: {
    introEnd: number
    firstDrop?: number | null
    safeEntries: number[]
    safeExits: number[]
    phraseBoundaries: number[]
  }
  embeddingIndexed: boolean
}

export interface Preparation {
  id: string
  sourcePath: string
  label: string
  status: JobStatus
  progress: number
  message: string
  trackCount: number
  discoveredTrackCount: number
  analysedTrackCount: number
  failedTrackCount: number
  currentTrack?: string | null
  genres: string[]
  tracks: Track[]
  modelReport?: {
    structureModel: string
    embeddingModel: string
    genreSource: string
    qdrantCollection: string
  } | null
  createdAt: string
  updatedAt: string
}

export interface MixOptions {
  preparationId: string
  genreOrder: string[]
  minTrackSeconds: number
  maxTrackSeconds: number
  acceptancePercentage: number
}

export interface Transition {
  fromTrackId: string
  toTrackId: string
  exitAtSeconds: number
  enterAtSeconds: number
  overlapSeconds: number
  bpmRatio: number
  tempoFactor: number
  loopSeconds: number
  beatMatched: boolean
  barMatched: boolean
  phraseMatched: boolean
  beatAlignmentErrorMs: number
  overlapBars: number
  fadeShape?: string
  spectrumPlan: string
  technique: string
  bassSwapProgress: number
  vocalClashRisk: number
  fadeOutCurve: number
  fadeInCurve: number
  outgoingLowDb: number
  incomingLowDb: number
  outgoingMidDb: number
  incomingMidDb: number
  outgoingHighDb: number
  incomingHighDb: number
  timingScore: number
  policyVersion: string
  style: string
  qualityScore: number
  renderQualityScore?: number | null
  notes: string[]
}

export interface SkipPlan {
  transition: Transition
  audioUrl: string
  resumeAtSeconds: number
}

export interface PlaylistItem {
  track: Track
  startSeconds: number
  endSeconds: number
  sourceStartSeconds?: number
  sourceEndSeconds?: number
  deckBpm?: number
  gainDb?: number
  transitionIn?: Transition | null
}

export interface Mix {
  id: string
  name: string
  status: JobStatus
  progress: number
  message: string
  options: MixOptions
  playlist: PlaylistItem[]
  rejectedTrackIds: string[]
  durationSeconds: number
  audioUrl?: string | null
  createdAt: string
  updatedAt: string
}

export interface Overview {
  preparations: Preparation[]
  mixes: Mix[]
  activeJobs: number
}
