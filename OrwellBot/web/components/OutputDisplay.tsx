"use client";

interface GenerationResult {
  prompt: string;
  response: string;
  tokens_generated: number;
}

interface OutputDisplayProps {
  result: GenerationResult;
}

export default function OutputDisplay({ result }: OutputDisplayProps) {
  const copyToClipboard = () => {
    const fullText = result.prompt + result.response;
    navigator.clipboard.writeText(fullText);
  };

  return (
    <section className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-medium text-orwell-muted uppercase tracking-wider">
          Generated text
        </h2>
        <button
          onClick={copyToClipboard}
          className="text-sm text-orwell-muted hover:text-orwell-ink transition-colors flex items-center gap-1"
          title="Copy to clipboard"
        >
          <CopyIcon />
          <span>Copy</span>
        </button>
      </div>

      <div className="p-6 bg-white rounded-lg border border-orwell-ink/10 shadow-sm">
        <p className="text-lg leading-relaxed font-serif text-orwell-ink whitespace-pre-wrap">
          {/* Show prompt in slightly muted color */}
          <span className="text-orwell-muted">{result.prompt}</span>
          {/* Show generated text in full color with typewriter effect */}
          <span className="typewriter-text">{result.response}</span>
        </p>
      </div>

      <p className="text-xs text-orwell-muted text-right">
        {result.tokens_generated} tokens generated
      </p>
    </section>
  );
}

function CopyIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  );
}
