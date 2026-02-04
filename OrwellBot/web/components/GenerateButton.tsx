"use client";

interface GenerateButtonProps {
  onClick: () => void;
  disabled?: boolean;
  isLoading?: boolean;
}

export default function GenerateButton({
  onClick,
  disabled = false,
  isLoading = false,
}: GenerateButtonProps) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`
        w-full py-4 px-6 rounded-lg
        font-serif text-lg font-medium
        transition-all duration-200
        flex items-center justify-center gap-3
        ${
          disabled
            ? "bg-orwell-paper text-orwell-muted cursor-not-allowed border border-orwell-ink/10"
            : "bg-orwell-ink text-orwell-cream hover:bg-orwell-ink/90 active:scale-[0.99] shadow-sm hover:shadow"
        }
      `}
    >
      {isLoading ? (
        <>
          <LoadingSpinner />
          <span>Generating...</span>
        </>
      ) : (
        <span>Generate</span>
      )}
    </button>
  );
}

function LoadingSpinner() {
  return (
    <svg
      className="spinner w-5 h-5"
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="3"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  );
}
