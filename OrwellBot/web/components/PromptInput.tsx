"use client";

import { KeyboardEvent } from "react";

interface PromptInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  disabled?: boolean;
}

export default function PromptInput({
  value,
  onChange,
  onSubmit,
  disabled = false,
}: PromptInputProps) {
  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Submit on Cmd+Enter (Mac) or Ctrl+Enter (Windows/Linux)
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      if (!disabled && value.trim()) {
        onSubmit();
      }
    }
  };

  return (
    <div className="relative">
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        placeholder="Enter a prompt for OrwellBot to continue..."
        rows={4}
        className={`
          w-full p-4 rounded-lg border-2
          bg-white text-orwell-ink text-lg font-serif
          placeholder:text-orwell-muted/60
          focus:outline-none focus:border-orwell-accent focus:ring-1 focus:ring-orwell-accent/20
          disabled:bg-orwell-paper disabled:cursor-not-allowed
          transition-colors resize-none
          ${disabled ? "border-orwell-ink/10" : "border-orwell-ink/20"}
        `}
      />
      <div className="absolute bottom-3 right-3 text-xs text-orwell-muted">
        {value.length > 0 && (
          <span className="opacity-60">
            {navigator.platform.includes("Mac") ? "⌘" : "Ctrl"}+Enter to
            generate
          </span>
        )}
      </div>
    </div>
  );
}
