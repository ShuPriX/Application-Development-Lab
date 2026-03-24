import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { TrendingUp } from "lucide-react";
import type { SentimentTrendPoint } from "../../types/analysis";

interface SentimentTrendProps {
  data: SentimentTrendPoint[];
}

export default function SentimentTrend({ data }: SentimentTrendProps) {
  if (data.length === 0) return null;

  return (
    <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-5 sm:p-6 border border-white/5">
      <h3 className="flex items-center gap-2 text-white font-semibold text-sm sm:text-base mb-4">
        <TrendingUp className="w-4 h-4 sm:w-5 sm:h-5 text-netflix-red" />
        Sentiment Over Time
      </h3>
      <ResponsiveContainer width="100%" height={250} minHeight={200}>
        <LineChart data={data}>
          <XAxis dataKey="date" tick={{ fill: "#6b7280", fontSize: 11 }} axisLine={{ stroke: "#333" }} />
          <YAxis tick={{ fill: "#6b7280", fontSize: 11 }} axisLine={{ stroke: "#333" }} />
          <Tooltip contentStyle={{ backgroundColor: "#1a1a1a", border: "1px solid #333", borderRadius: "8px", color: "#fff", fontSize: 13 }} />
          <Legend wrapperStyle={{ color: "#9ca3af", fontSize: 13 }} />
          <Line type="monotone" dataKey="positive" stroke="#22c55e" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="neutral" stroke="#eab308" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="negative" stroke="#ef4444" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
