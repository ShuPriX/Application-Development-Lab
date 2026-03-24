import { useState, useEffect, useRef, useCallback } from "react";
import { AnimatePresence, motion } from "framer-motion";

interface PlaceholderInputProps {
  placeholders: string[];
  value: string;
  onChange: (value: string) => void;
  onFocus?: () => void;
  onBlur?: () => void;
}

export default function PlaceholderInput({
  placeholders,
  value,
  onChange,
  onFocus,
  onBlur,
}: PlaceholderInputProps) {
  const [placeholderIndex, setPlaceholderIndex] = useState(0);
  const [isFocused, setIsFocused] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isFocused || value) return;
    const interval = setInterval(() => {
      setPlaceholderIndex((prev) => (prev + 1) % placeholders.length);
    }, 3000);
    return () => clearInterval(interval);
  }, [placeholders.length, isFocused, value]);

  const handleFocus = useCallback(() => {
    setIsFocused(true);
    onFocus?.();
  }, [onFocus]);

  const handleBlur = useCallback(() => {
    setIsFocused(false);
    onBlur?.();
  }, [onBlur]);

  return (
    <div
      className="relative w-full max-w-xl mx-auto group cursor-text"
      onClick={() => inputRef.current?.focus()}
    >
      <div className="relative flex items-center rounded-xl border border-white/10 bg-white/5 backdrop-blur-md px-4 py-3 transition-all duration-300 hover:border-white/20 focus-within:border-netflix-red focus-within:bg-white/[0.07] focus-within:shadow-[0_0_30px_-5px_rgba(229,9,20,0.3)]">
        {/* Search icon */}
        <svg
          className="w-5 h-5 text-gray-500 mr-3 flex-shrink-0"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
          />
        </svg>

        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onFocus={handleFocus}
          onBlur={handleBlur}
          className="flex-1 bg-transparent outline-none text-white text-base placeholder-transparent"
          placeholder=""
        />

        {/* Animated placeholder */}
        {!value && !isFocused && (
          <div className="absolute left-12 top-1/2 -translate-y-1/2 pointer-events-none">
            <AnimatePresence mode="wait">
              <motion.span
                key={placeholders[placeholderIndex]}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 0.5, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.3 }}
                className="text-gray-500 text-base whitespace-nowrap"
              >
                {placeholders[placeholderIndex]}
              </motion.span>
            </AnimatePresence>
          </div>
        )}

        {/* Static placeholder when focused but empty */}
        {!value && isFocused && (
          <span className="absolute left-12 top-1/2 -translate-y-1/2 text-gray-600 text-base pointer-events-none">
            Type a movie name...
          </span>
        )}
      </div>
    </div>
  );
}
