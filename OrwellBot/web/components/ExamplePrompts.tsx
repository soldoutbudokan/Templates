"use client";

interface ExamplePromptsProps {
  onSelect: (prompt: string) => void;
}

const EXAMPLES = [
  "The greatest enemy of clear language is",
  "In our time, political speech consists largely of",
  "Power is not a means; it is",
  "If liberty means anything at all, it means",
  "The past was erased, the erasure was forgotten,",
  "All animals are equal, but",
];

export default function ExamplePrompts({ onSelect }: ExamplePromptsProps) {
  return (
    <div className="flex flex-wrap gap-2">
      {EXAMPLES.map((example, index) => (
        <button
          key={index}
          onClick={() => onSelect(example)}
          className="
            px-3 py-1.5 rounded-full
            text-sm font-serif
            bg-orwell-paper text-orwell-muted
            border border-orwell-ink/10
            hover:bg-orwell-ink hover:text-orwell-cream
            hover:border-orwell-ink
            transition-colors duration-200
            truncate max-w-[280px]
          "
          title={example}
        >
          {example}
        </button>
      ))}
    </div>
  );
}
