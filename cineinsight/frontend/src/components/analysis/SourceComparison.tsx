import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { Scale } from "lucide-react";
import type { SourceComparison as SourceComparisonType } from "../../types/analysis";

interface SourceComparisonProps {
  data: SourceComparisonType;
}

const SOURCE_LABELS: Record<string, string> = {
  tmdb: "TMDB",
  imdb: "IMDB",
  rottentomatoes: "Rotten Tomatoes",
};

export default function SourceComparison({ data }: SourceComparisonProps) {
  const chartData = Object.entries(data)
    .filter(([, dist]) => dist && (dist.positive + dist.neutral + dist.negative) > 0)
    .map(([key, dist]) => ({
      source: SOURCE_LABELS[key] || key,
      Positive: dist!.positive,
      Neutral: dist!.neutral,
      Negative: dist!.negative,
    }));

  return (
    <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-5 sm:p-6 border border-white/5">
      <h3 className="flex items-center gap-2 text-white font-semibold text-sm sm:text-base mb-4">
        <Scale className="w-4 h-4 sm:w-5 sm:h-5 text-netflix-red" />
        Source Comparison
      </h3>
      <ResponsiveContainer width="100%" height={250} minHeight={200}>
        <BarChart data={chartData}>
          <XAxis dataKey="source" tick={{ fill: "#9ca3af", fontSize: 13 }} axisLine={{ stroke: "#333" }} />
          <YAxis tick={{ fill: "#6b7280", fontSize: 11 }} axisLine={{ stroke: "#333" }} />
          <Tooltip contentStyle={{ backgroundColor: "#1a1a1a", border: "1px solid #333", borderRadius: "8px", color: "#fff", fontSize: 13 }} />
          <Legend wrapperStyle={{ color: "#9ca3af", fontSize: 13 }} />
          <Bar dataKey="Positive" fill="#22c55e" radius={[4, 4, 0, 0]} />
          <Bar dataKey="Neutral" fill="#eab308" radius={[4, 4, 0, 0]} />
          <Bar dataKey="Negative" fill="#ef4444" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
