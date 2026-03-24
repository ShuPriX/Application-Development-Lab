import { useEffect, useState } from "react";
import { motion } from "framer-motion";

interface TextGenerateEffectProps {
  words: string;
  className?: string;
  delay?: number;
}

export default function TextGenerateEffect({
  words,
  className = "",
  delay = 0,
}: TextGenerateEffectProps) {
  const [started, setStarted] = useState(false);
  const wordArray = words.split(" ");

  useEffect(() => {
    const timeout = setTimeout(() => setStarted(true), delay);
    return () => clearTimeout(timeout);
  }, [delay]);

  if (!started) {
    return <p className={className}>&nbsp;</p>;
  }

  return (
    <p className={className}>
      {wordArray.map((word, idx) => (
        <motion.span
          key={`${word}-${idx}`}
          initial={{ opacity: 0, filter: "blur(4px)" }}
          animate={{ opacity: 1, filter: "blur(0px)" }}
          transition={{
            duration: 0.3,
            delay: idx * 0.06,
            ease: "easeOut",
          }}
          className="inline-block mr-[0.3em]"
        >
          {word}
        </motion.span>
      ))}
    </p>
  );
}
