/**
 * Material Symbols Outlined icon wrapper.
 * Uses the locally-hosted MaterialSymbolsOutlined.woff2 font
 * defined in globals.css via the .material-symbols-outlined class.
 *
 * Usage: <Icon name="smart_toy" size={18} className="text-white" />
 */

interface IconProps {
  /** Material Symbols icon name, e.g. "smart_toy", "settings" */
  name: string;
  /** Icon font-size in px (default: 24) */
  size?: number;
  /** Additional CSS classes */
  className?: string;
  /** aria-hidden (default: true) */
  ariaHidden?: boolean;
}

export function Icon({
  name,
  size,
  className = "",
  ariaHidden = true,
}: IconProps) {
  return (
    <span
      className={`material-symbols-outlined ${className}`}
      style={size ? { fontSize: `${size}px` } : undefined}
      aria-hidden={ariaHidden}
    >
      {name}
    </span>
  );
}
