import { useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Progress } from "@heroui/react";
import {
  Terminal,
  Search,
  Brain,
  Cpu,
  CheckCircle,
  XCircle,
  AlertCircle,
  Database,
  Sparkles,
  MessageSquareText,
  ThumbsUp,
  ThumbsDown,
  Minus,
} from "lucide-react";
import type { AnalysisProgress } from "../../types/analysis";
import type { AnalysisLiveState } from "../../hooks/useAnalysis";

const STAGE_ICONS: Record<string, React.ReactNode> = {
  metadata: <Database className="w-3.5 h-3.5" />,
  scraping: <Search className="w-3.5 h-3.5" />,
  storing: <Database className="w-3.5 h-3.5" />,
  bert: <Brain className="w-3.5 h-3.5" />,
  bilstm: <Cpu className="w-3.5 h-3.5" />,
  aggregation: <Sparkles className="w-3.5 h-3.5" />,
  saving: <Database className="w-3.5 h-3.5" />,
  complete: <CheckCircle className="w-3.5 h-3.5" />,
};

const STAGE_COLORS: Record<string, string> = {
  metadata: "text-blue-400",
  scraping: "text-amber-400",
  storing: "text-cyan-400",
  bert: "text-purple-400",
  bilstm: "text-pink-400",
  aggregation: "text-emerald-400",
  saving: "text-cyan-400",
  complete: "text-green-400",
};

const SENTIMENT_CONFIG = {
  positive: { icon: ThumbsUp, color: "text-green-400", bg: "bg-green-400/10" },
  negative: { icon: ThumbsDown, color: "text-red-400", bg: "bg-red-400/10" },
  neutral: { icon: Minus, color: "text-yellow-400", bg: "bg-yellow-400/10" },
};

const SOURCE_STATUS_ICON = {
  done: <CheckCircle className="w-3.5 h-3.5 text-green-400" />,
  error: <XCircle className="w-3.5 h-3.5 text-red-400" />,
  empty: <AlertCircle className="w-3.5 h-3.5 text-yellow-400" />,
  skipped: <AlertCircle className="w-3.5 h-3.5 text-gray-500" />,
};

function LogLine({ entry, index }: { entry: AnalysisProgress; index: number }) {
  const stageColor = STAGE_COLORS[entry.stage] || "text-gray-400";
  const icon = STAGE_ICONS[entry.stage] || <Terminal className="w-3.5 h-3.5" />;

  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.25 }}
      className="flex items-start gap-2 py-1 font-mono text-xs sm:text-sm"
    >
      <span className="text-gray-600 select-none shrink-0 w-6 text-right">
        {String(index + 1).padStart(2, "0")}
      </span>
      <span className={`shrink-0 mt-0.5 ${stageColor}`}>{icon}</span>
      <span className={`shrink-0 ${stageColor}`}>[{entry.stage}]</span>
      <span className="text-gray-300">{entry.message}</span>
    </motion.div>
  );
}

function SourceCard({
  entry,
}: {
  entry: AnalysisProgress;
}) {
  const statusIcon = SOURCE_STATUS_ICON[(entry.status || "done") as keyof typeof SOURCE_STATUS_ICON];
  const sourceLabel =
    entry.source === "tmdb"
      ? "TMDB"
      : entry.source === "imdb"
        ? "IMDB"
        : entry.source === "rottentomatoes"
          ? "Rotten Tomatoes"
          : entry.source;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className="flex items-center gap-2 bg-white/5 rounded-lg px-3 py-2 border border-white/5"
    >
      {statusIcon}
      <span className="text-gray-300 text-xs sm:text-sm font-medium">{sourceLabel}</span>
      <span className="text-gray-500 text-xs ml-auto">
        {entry.count} review{entry.count !== 1 ? "s" : ""}
      </span>
    </motion.div>
  );
}

function ReviewPreview({
  review,
  index,
}: {
  review: { source: string; author: string; snippet: string };
  index: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.06 }}
      className="bg-white/[0.03] rounded-lg px-3 py-2 border border-white/5"
    >
      <div className="flex items-center gap-2 mb-1">
        <MessageSquareText className="w-3 h-3 text-gray-500" />
        <span className="text-gray-500 text-[10px] uppercase tracking-wider">
          {review.source}
        </span>
        <span className="text-gray-600 text-[10px]">— {review.author}</span>
      </div>
      <p className="text-gray-400 text-xs leading-relaxed">{review.snippet}</p>
    </motion.div>
  );
}

function SentimentPreview({
  prediction,
  index,
}: {
  prediction: { snippet: string; source: string; label: string; score: number };
  index: number;
}) {
  const config = SENTIMENT_CONFIG[prediction.label as keyof typeof SENTIMENT_CONFIG] || SENTIMENT_CONFIG.neutral;
  const Icon = config.icon;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.06 }}
      className={`rounded-lg px-3 py-2 border border-white/5 ${config.bg}`}
    >
      <div className="flex items-center gap-2 mb-1">
        <Icon className={`w-3.5 h-3.5 ${config.color}`} />
        <span className={`text-xs font-semibold ${config.color}`}>
          {prediction.label}
        </span>
        <span className="text-gray-500 text-[10px] ml-auto">
          {(prediction.score * 100).toFixed(1)}% conf
        </span>
      </div>
      <p className="text-gray-400 text-xs leading-relaxed">{prediction.snippet}</p>
    </motion.div>
  );
}

function VerdictReveal({ entry }: { entry: AnalysisProgress }) {
  const verdictColors: Record<string, string> = {
    "Strongly Recommended": "text-green-400",
    "Worth Watching": "text-emerald-400",
    "Mixed Reviews": "text-yellow-400",
    "Below Average": "text-orange-400",
    "Skip It": "text-red-400",
  };
  const color = verdictColors[entry.verdict || ""] || "text-gray-400";

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ type: "spring", duration: 0.6 }}
      className="flex items-center justify-center gap-3 bg-white/5 rounded-xl px-4 py-3 border border-white/10"
    >
      <Sparkles className={`w-5 h-5 ${color}`} />
      <span className={`text-lg sm:text-xl font-bold ${color}`}>
        {entry.verdict}
      </span>
      {entry.confidence != null && (
        <span className="text-gray-500 text-sm">
          ({(entry.confidence * 100).toFixed(1)}%)
        </span>
      )}
    </motion.div>
  );
}

function AspectBars({ aspects }: { aspects: Record<string, number> }) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
      {Object.entries(aspects).map(([aspect, score], i) => (
        <motion.div
          key={aspect}
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: i * 0.08 }}
          className="flex items-center gap-2"
        >
          <span className="text-gray-400 text-xs capitalize w-16 shrink-0">
            {aspect}
          </span>
          <div className="flex-1 h-2 bg-white/5 rounded-full overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${(score as number) * 100}%` }}
              transition={{ duration: 0.6, delay: i * 0.08 }}
              className="h-full bg-netflix-red rounded-full"
            />
          </div>
          <span className="text-gray-500 text-xs w-8 text-right">
            {((score as number) * 100).toFixed(0)}%
          </span>
        </motion.div>
      ))}
    </div>
  );
}

export default function LiveAnalysis({
  liveState,
  movieTitle,
  posterUrl,
}: {
  liveState: AnalysisLiveState;
  movieTitle?: string;
  posterUrl?: string | null;
}) {
  const { latest, logs } = liveState;
  const logContainerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll only within the terminal container
  useEffect(() => {
    const el = logContainerRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [logs.length]);

  const sourceEntries = logs.filter((l) => l.type === "source");
  const reviewEntries = logs.filter((l) => l.type === "reviews");
  const sentimentEntries = logs.filter((l) => l.type === "sentiments");
  const aspectEntries = logs.filter((l) => l.type === "aspects");
  const verdictEntries = logs.filter((l) => l.type === "verdict");
  const logEntries = logs.filter(
    (l) => l.type === "log" || l.type === "source",
  );

  return (
    <div className="w-full max-w-4xl mx-auto space-y-4">
      {/* Header */}
      <div className="flex items-center gap-4">
        {posterUrl && (
          <img
            src={posterUrl}
            alt=""
            className="w-14 h-20 rounded-lg object-cover shadow-lg"
          />
        )}
        <div className="flex-1 min-w-0">
          <h2 className="text-white text-lg sm:text-xl font-bold truncate">
            Analyzing {movieTitle || "Movie"}
          </h2>
          <p className="text-gray-500 text-xs sm:text-sm">
            {latest?.message || "Starting analysis..."}
          </p>
        </div>
      </div>

      {/* Progress bar */}
      <Progress
        value={latest?.progress || 0}
        className="w-full"
        classNames={{
          indicator: "bg-netflix-red",
          track: "bg-white/10",
        }}
      />

      {/* Terminal log */}
      <div className="bg-black/40 rounded-xl border border-white/5 overflow-hidden">
        <div className="flex items-center gap-2 px-4 py-2 border-b border-white/5 bg-white/[0.02]">
          <Terminal className="w-4 h-4 text-gray-500" />
          <span className="text-gray-500 text-xs font-mono">
            analysis pipeline
          </span>
          <div className="flex gap-1.5 ml-auto">
            <div className="w-2.5 h-2.5 rounded-full bg-red-500/60" />
            <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/60" />
            <div className="w-2.5 h-2.5 rounded-full bg-green-500/60" />
          </div>
        </div>
        <div ref={logContainerRef} className="px-4 py-3 max-h-[200px] overflow-y-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-white/10">
          <AnimatePresence>
            {logEntries.map((entry, i) => (
              <LogLine key={i} entry={entry} index={i} />
            ))}
          </AnimatePresence>
        </div>
      </div>

      {/* Source status cards */}
      <AnimatePresence>
        {sourceEntries.length > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="grid grid-cols-3 gap-2"
          >
            {sourceEntries.map((entry, i) => (
              <SourceCard key={i} entry={entry} />
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Review previews */}
      <AnimatePresence>
        {reviewEntries.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <p className="text-gray-500 text-xs uppercase tracking-wider mb-2 flex items-center gap-1.5">
              <MessageSquareText className="w-3.5 h-3.5" />
              Scraped Reviews
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-h-[200px] overflow-y-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-white/10">
              {reviewEntries[reviewEntries.length - 1].reviews?.map((r, i) => (
                <ReviewPreview key={i} review={r} index={i} />
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Sentiment predictions */}
      <AnimatePresence>
        {sentimentEntries.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <p className="text-gray-500 text-xs uppercase tracking-wider mb-2 flex items-center gap-1.5">
              <Brain className="w-3.5 h-3.5" />
              BERT Predictions
            </p>
            {/* Counts */}
            {sentimentEntries[sentimentEntries.length - 1].counts && (
              <div className="flex gap-3 mb-2">
                {Object.entries(
                  sentimentEntries[sentimentEntries.length - 1].counts!,
                ).map(([label, count]) => {
                  const cfg = SENTIMENT_CONFIG[label as keyof typeof SENTIMENT_CONFIG];
                  return (
                    <motion.div
                      key={label}
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      className={`flex items-center gap-1.5 ${cfg.bg} rounded-lg px-3 py-1.5`}
                    >
                      <cfg.icon className={`w-3.5 h-3.5 ${cfg.color}`} />
                      <span className={`text-sm font-bold ${cfg.color}`}>
                        {count}
                      </span>
                      <span className="text-gray-500 text-xs">{label}</span>
                    </motion.div>
                  );
                })}
              </div>
            )}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-h-[200px] overflow-y-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-white/10">
              {sentimentEntries[sentimentEntries.length - 1].predictions?.map(
                (p, i) => (
                  <SentimentPreview key={i} prediction={p} index={i} />
                ),
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Aspect scores */}
      <AnimatePresence>
        {aspectEntries.length > 0 && aspectEntries[aspectEntries.length - 1].aspects && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <p className="text-gray-500 text-xs uppercase tracking-wider mb-2 flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5" />
              BiLSTM Aspect Scores
            </p>
            <AspectBars aspects={aspectEntries[aspectEntries.length - 1].aspects!} />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Verdict reveal */}
      <AnimatePresence>
        {verdictEntries.length > 0 && (
          <VerdictReveal entry={verdictEntries[verdictEntries.length - 1]} />
        )}
      </AnimatePresence>
    </div>
  );
}
