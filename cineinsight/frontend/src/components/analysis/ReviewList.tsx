import { useState } from "react";
import { Chip, Pagination } from "@heroui/react";
import { motion } from "framer-motion";
import { MessageSquare, User, Globe } from "lucide-react";
import type { ReviewSentiment } from "../../types/analysis";

interface ReviewListProps {
  reviews: ReviewSentiment[];
}

const sentimentColors = {
  positive: "success",
  negative: "danger",
  neutral: "warning",
} as const;

const ITEMS_PER_PAGE = 8;

export default function ReviewList({ reviews }: ReviewListProps) {
  const [page, setPage] = useState(1);
  const totalPages = Math.ceil(reviews.length / ITEMS_PER_PAGE);
  const displayed = reviews.slice((page - 1) * ITEMS_PER_PAGE, page * ITEMS_PER_PAGE);

  return (
    <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-5 sm:p-6 border border-white/5">
      <h3 className="flex items-center gap-2 text-white font-semibold text-sm sm:text-base mb-5">
        <MessageSquare className="w-4 h-4 sm:w-5 sm:h-5 text-netflix-red" />
        Individual Reviews ({reviews.length})
      </h3>
      <div className="space-y-3">
        {displayed.map((review, i) => (
          <motion.div
            key={review.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.05 }}
            className="bg-white/[0.03] rounded-xl p-4 sm:p-5 border border-white/5"
          >
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="flex items-center gap-1.5 text-gray-300 text-sm sm:text-base font-medium">
                  <User className="w-4 h-4 text-gray-500" />
                  {review.author || "Anonymous"}
                </span>
                <Chip size="sm" variant="flat" className="text-xs capitalize" color={sentimentColors[review.sentiment_label]}>
                  {review.sentiment_label}
                </Chip>
                <span className="text-gray-600 text-xs sm:text-sm">
                  {Math.round(review.sentiment_score * 100)}%
                </span>
              </div>
              <Chip size="sm" variant="bordered" className="text-xs text-gray-400 border-gray-700" startContent={<Globe className="w-3 h-3" />}>
                {review.source}
              </Chip>
            </div>
            <p className="text-gray-400 text-sm sm:text-base leading-relaxed line-clamp-3">
              {review.content}
            </p>
          </motion.div>
        ))}
      </div>
      {totalPages > 1 && (
        <div className="flex justify-center mt-6">
          <Pagination total={totalPages} page={page} onChange={setPage} size="sm" classNames={{ cursor: "bg-netflix-red" }} />
        </div>
      )}
    </div>
  );
}
