type IconName = 'cards' | 'spark' | 'target' | 'flame' | 'trophy' | 'check' | 'arrow' | 'pause' | 'clock';

const paths: Record<IconName, string> = {
  cards: 'M8 3h11a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2ZM3 6l-1 12a2 2 0 0 0 2 2M13.5 8l3 4-3 4-3-4 3-4Z',
  spark: 'm12 2 2.7 7.3L22 12l-7.3 2.7L12 22l-2.7-7.3L2 12l7.3-2.7L12 2Z',
  target: 'M21 12a9 9 0 1 1-9-9M17 12a5 5 0 1 1-5-5M12 12l8-8M16 4h4v4',
  flame: 'M13 2c1 5-3 6-1 10 1-1 3-2 3-5 4 4 5 7 3 11-3 5-10 5-13 0-2-4 0-8 3-11 0 4 2 5 2 5-1-4 3-6 3-10Z',
  trophy: 'M7 3h10v6a5 5 0 0 1-10 0V3ZM7 5H3v3a4 4 0 0 0 4 4M17 5h4v3a4 4 0 0 1-4 4M12 14v5M7 21h10M9 19h6',
  check: 'm5 12 4 4L19 6',
  arrow: 'M4 12h15M13 6l6 6-6 6',
  pause: 'M8 5v14M16 5v14',
  clock: 'M12 8v5l3 2M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z',
};

export default function GameIcon({ name, className = '' }: { name: IconName; className?: string }) {
  return <svg className={`game-icon ${className}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.65" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false"><path d={paths[name]} /></svg>;
}
