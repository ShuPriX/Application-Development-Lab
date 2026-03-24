import { motion } from "framer-motion";
import { ThumbsUp, CheckCircle, HelpCircle, ThumbsDown, XCircle } from "lucide-react";

interface VerdictBadgeProps {
  verdict: string;
  confidence: number;
  sentiment: string;
}

const verdictConfig: Record<string, { color: string; bg: string; glow: string; icon: typeof ThumbsUp }> = {
  "Strongly Recommended": { color: "text-green-400", bg: "bg-green-500/10", glow: "shadow-green-500/20", icon: ThumbsUp },
  "Worth Watching": { color: "text-emerald-400", bg: "bg-emerald-500/10", glow: "shadow-emerald-500/20", icon: CheckCircle },
  "Mixed Reviews": { color: "text-yellow-400", bg: "bg-yellow-500/10", glow: "shadow-yellow-500/20", icon: HelpCircle },
  "Below Average": { color: "text-orange-400", bg: "bg-orange-500/10", glow: "shadow-orange-500/20", icon: ThumbsDown },
  "Skip It": { color: "text-red-400", bg: "bg-red-500/10", glow: "shadow-red-500/20", icon: XCircle },
};

export default function VerdictBadge({ verdict, confidence, sentiment }: VerdictBadgeProps) {
  const config = verdictConfig[verdict] || verdictConfig["Mixed Reviews"];
  const Icon = config.icon;

  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ delay: 0.3, type: "spring", stiffness: 200 }}
      className={`${config.bg} ${config.glow} shadow-lg rounded-2xl p-5 sm:p-6 backdrop-blur-sm border border-white/5 text-center`}
    >
      <p className="text-gray-400 text-xs uppercase tracking-widest mb-3">Verdict</p>
      <Icon className={`w-10 h-10 sm:w-12 sm:h-12 mx-auto mb-3 ${config.color}`} />
      <h2 className={`text-2xl sm:text-3xl font-extrabold ${config.color} mb-2`}>{verdict}</h2>
      <p className="text-gray-400 text-sm sm:text-base">
        Overall: <span className="text-white capitalize">{sentiment}</span>
      </p>
      <div className="mt-4">
        <div className="flex justify-between text-xs sm:text-sm text-gray-500 mb-1">
          <span>Confidence</span>
          <span>{Math.round(confidence * 100)}%</span>
        </div>
        <div className="h-2 bg-white/5 rounded-full overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${confidence * 100}%` }}
            transition={{ delay: 0.5, duration: 0.8 }}
            className={`h-full rounded-full ${confidence > 0.8 ? "bg-green-500" : confidence > 0.6 ? "bg-yellow-500" : "bg-red-500"}`}
          />
        </div>
      </div>
    </motion.div>
  );
}
