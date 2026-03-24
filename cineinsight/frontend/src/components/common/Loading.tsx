import { Loader2 } from "lucide-react";

interface LoadingProps {
  message?: string;
}

export default function Loading({ message = "Loading..." }: LoadingProps) {
  return (
    <div className="flex flex-col items-center justify-center py-20 gap-4 min-h-[50vh]">
      <Loader2 className="w-10 h-10 text-netflix-red animate-spin" />
      <p className="text-gray-400 text-base">{message}</p>
    </div>
  );
}
