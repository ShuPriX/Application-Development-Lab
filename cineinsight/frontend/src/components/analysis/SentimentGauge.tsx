import { motion } from "framer-motion";
import { SmilePlus, Meh, Frown, BarChart3 } from "lucide-react";
import type { SentimentDistribution } from "../../types/analysis";

interface SentimentGaugeProps {
  distribution: SentimentDistribution;
}

export default function SentimentGauge({ distribution }: SentimentGaugeProps) {
  const total = distribution.positive + distribution.negative + distribution.neutral;
  const pctPositive = total > 0 ? (distribution.positive / total) * 100 : 0;
  const pctNeutral = total > 0 ? (distribution.neutral / total) * 100 : 0;
  const pctNegative = total > 0 ? (distribution.negative / total) * 100 : 0;

  return (
    <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-5 sm:p-6 border border-white/5">
      <h3 className="flex items-center gap-2 text-white font-semibold text-sm sm:text-base mb-5">
        <BarChart3 className="w-4 h-4 sm:w-5 sm:h-5 text-netflix-red" />
        Sentiment Distribution
      </h3>
      <div className="space-y-5">
        <SentimentBar label="Positive" icon={<SmilePlus className="w-4 h-4 text-green-400" />} value={distribution.positive} percentage={pctPositive} color="bg-green-500" />
        <SentimentBar label="Neutral" icon={<Meh className="w-4 h-4 text-yellow-400" />} value={distribution.neutral} percentage={pctNeutral} color="bg-yellow-500" />
        <SentimentBar label="Negative" icon={<Frown className="w-4 h-4 text-red-400" />} value={distribution.negative} percentage={pctNegative} color="bg-red-500" />
      </div>
      <p className="text-gray-500 text-xs sm:text-sm mt-5">{total} reviews total</p>
    </div>
  );
}

function SentimentBar({ label, icon, value, percentage, color }: { label: string; icon: React.ReactNode; value: number; percentage: number; color: string }) {
  return (
    <div>
      <div className="flex justify-between text-xs sm:text-sm mb-1.5">
        <span className="flex items-center gap-1.5 text-gray-300">{icon} {label}</span>
        <span className="text-gray-400">{value} ({Math.round(percentage)}%)</span>
      </div>
      <div className="h-2.5 bg-white/5 rounded-full overflow-hidden">
        <motion.div initial={{ width: 0 }} animate={{ width: `${percentage}%` }} transition={{ duration: 0.6, delay: 0.2 }} className={`h-full rounded-full ${color}`} />
      </div>
    </div>
  );
}
