import { useMemo } from "react";
import { Cloud } from "lucide-react";
import { motion } from "framer-motion";
import type { WordCloudItem } from "../../types/analysis";

interface WordCloudProps {
  positive: WordCloudItem[];
  negative: WordCloudItem[];
}

function WordCloudDisplay({
  words,
  label,
  colorClass,
  colorFrom,
  colorTo,
}: {
  words: WordCloudItem[];
  label: string;
  colorClass: string;
  colorFrom: string;
  colorTo: string;
}) {
  const sorted = useMemo(() => {
    if (!words.length) return [];
    const maxVal = Math.max(...words.map((w) => w.value), 1);
    // Sort by value descending so big words come first
    return words
      .slice(0, 35)
      .sort((a, b) => b.value - a.value)
      .map((w, i) => {
        const norm = w.value / maxVal;
        return { ...w, norm, index: i };
      });
  }, [words]);

  if (!sorted.length) return null;

  return (
    <div>
      <p className={`text-sm sm:text-base font-medium mb-3 ${colorClass}`}>
        {label}
      </p>
      <div className="relative rounded-xl bg-black/30 border border-white/5 p-4 sm:p-5 min-h-[240px] flex items-center justify-center">
        <div className="flex flex-wrap items-center justify-center gap-x-2.5 gap-y-1.5">
          {sorted.map((w, i) => {
            const size = w.norm > 0.8
              ? "text-3xl sm:text-4xl"
              : w.norm > 0.6
                ? "text-2xl sm:text-3xl"
                : w.norm > 0.4
                  ? "text-xl sm:text-2xl"
                  : w.norm > 0.25
                    ? "text-base sm:text-lg"
                    : "text-xs sm:text-sm";

            const weight = w.norm > 0.6 ? "font-bold" : w.norm > 0.3 ? "font-semibold" : "font-normal";
            const opacity = 0.4 + w.norm * 0.6;

            return (
              <motion.span
                key={w.text}
                initial={{ opacity: 0, scale: 0.6 }}
                animate={{ opacity, scale: 1 }}
                transition={{ duration: 0.4, delay: i * 0.02 }}
                className={`${size} ${weight} select-none cursor-default transition-all duration-200 hover:!opacity-100 hover:scale-110`}
                style={{
                  background: `linear-gradient(135deg, ${colorFrom}, ${colorTo})`,
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent",
                  backgroundClip: "text",
                }}
              >
                {w.text}
              </motion.span>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export default function WordCloud({ positive, negative }: WordCloudProps) {
  if (!positive.length && !negative.length) return null;

  return (
    <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-5 sm:p-6 border border-white/5">
      <h3 className="flex items-center gap-2 text-white font-semibold text-sm sm:text-base mb-5">
        <Cloud className="w-4 h-4 sm:w-5 sm:h-5 text-netflix-red" />
        Word Clouds
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <WordCloudDisplay
          words={positive}
          label="Positive Reviews"
          colorClass="text-green-400"
          colorFrom="#4ade80"
          colorTo="#22c55e"
        />
        <WordCloudDisplay
          words={negative}
          label="Negative Reviews"
          colorClass="text-red-400"
          colorFrom="#f87171"
          colorTo="#ef4444"
        />
      </div>
    </div>
  );
}
