import { RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Tooltip } from "recharts";
import { Target } from "lucide-react";
import type { AspectScores } from "../../types/analysis";

interface AspectRadarProps {
  scores: AspectScores;
}

export default function AspectRadar({ scores }: AspectRadarProps) {
  const data = [
    { aspect: "Acting", score: scores.acting * 100 },
    { aspect: "Plot", score: scores.plot * 100 },
    { aspect: "Visuals", score: scores.visuals * 100 },
    { aspect: "Music", score: scores.music * 100 },
    { aspect: "Direction", score: scores.direction * 100 },
  ];

  return (
    <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-5 sm:p-6 border border-white/5">
      <h3 className="flex items-center gap-2 text-white font-semibold text-sm sm:text-base mb-4">
        <Target className="w-4 h-4 sm:w-5 sm:h-5 text-netflix-red" />
        Aspect Analysis
      </h3>
      <ResponsiveContainer width="100%" height={260} minHeight={220}>
        <RadarChart data={data}>
          <PolarGrid stroke="#333" />
          <PolarAngleAxis dataKey="aspect" tick={{ fill: "#9ca3af", fontSize: 13 }} />
          <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fill: "#6b7280", fontSize: 11 }} />
          <Tooltip contentStyle={{ backgroundColor: "#1a1a1a", border: "1px solid #333", borderRadius: "8px", color: "#fff", fontSize: 13 }} formatter={(value) => [`${Math.round(Number(value))}%`, "Score"]} />
          <Radar name="Score" dataKey="score" stroke="#E50914" fill="#E50914" fillOpacity={0.2} strokeWidth={2} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
