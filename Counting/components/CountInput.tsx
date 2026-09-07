'use client';

interface Props {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  autoFocus?: boolean;
}

export default function CountInput({ id, label, value, onChange, autoFocus }: Props) {
  return <div className="count-field">
    <label htmlFor={id}>{label}</label>
    <div className="count-input-row">
      <button type="button" className="sign-button" aria-label={`Change sign of ${label.toLowerCase()}`}
        onClick={() => onChange(value.startsWith('-') ? value.slice(1) : `-${value.replace(/^\+/, '')}`)}>±</button>
      <input id={id} value={value} onChange={event => onChange(event.target.value)}
        inputMode="numeric" autoComplete="off" autoFocus={autoFocus} placeholder="0" maxLength={5} />
    </div>
  </div>;
}
