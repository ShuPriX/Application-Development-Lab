import { ThumbsUp, ThumbsDown, FileText } from "lucide-react";

interface ReviewSummaryProps {
  positive: string[];
  negative: string[];
}

export default function ReviewSummary({ positive, negative }: ReviewSummaryProps) {
  if (!positive.length && !negative.length) return null;

  return (
    <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-5 sm:p-6 border border-white/5">
      <h3 className="flex items-center gap-2 text-white font-semibold text-sm sm:text-base mb-5">
        <FileText className="w-4 h-4 sm:w-5 sm:h-5 text-netflix-red" />
        Review Summary
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <p className="flex items-center gap-2 text-green-400 text-xs sm:text-sm uppercase tracking-wider mb-4 font-medium">
            <ThumbsUp className="w-4 h-4" />
            What people liked
          </p>
          <ul className="space-y-3">
            {positive.map((sentence, i) => (
              <li key={i} className="text-gray-300 text-sm sm:text-base pl-4 border-l-2 border-green-500/30 leading-relaxed">
                {sentence}
              </li>
            ))}
          </ul>
        </div>
        <div>
          <p className="flex items-center gap-2 text-red-400 text-xs sm:text-sm uppercase tracking-wider mb-4 font-medium">
            <ThumbsDown className="w-4 h-4" />
            Common criticisms
          </p>
          <ul className="space-y-3">
            {negative.map((sentence, i) => (
              <li key={i} className="text-gray-300 text-sm sm:text-base pl-4 border-l-2 border-red-500/30 leading-relaxed">
                {sentence}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
