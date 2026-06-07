import * as React from "react"
import * as SliderPrimitive from "@radix-ui/react-slider"

import { cn } from "@/lib/utils"

interface SliderProps {
  min?: number | string
  max?: number | string
  step?: number | string
  value?: number
  defaultValue?: number
  onChange?: (e: { target: { value: string } }) => void
  disabled?: boolean
  className?: string
}

const Slider = React.forwardRef<
  React.ElementRef<typeof SliderPrimitive.Root>,
  SliderProps
>(({ min = 0, max = 100, step = 1, value, defaultValue, onChange, disabled, className }, ref) => {
  return (
    <SliderPrimitive.Root
      ref={ref}
      min={Number(min)}
      max={Number(max)}
      step={Number(step)}
      value={value !== undefined ? [value] : undefined}
      defaultValue={defaultValue !== undefined ? [defaultValue] : undefined}
      onValueChange={([v]) => {
        onChange?.({ target: { value: String(v) } })
      }}
      disabled={disabled}
      className={cn(
        "relative flex w-full touch-none select-none items-center",
        className
      )}
    >
      <SliderPrimitive.Track className="relative h-1.5 w-full grow overflow-hidden rounded-full bg-primary/20">
        <SliderPrimitive.Range className="absolute h-full bg-primary" />
      </SliderPrimitive.Track>
      <SliderPrimitive.Thumb className="block h-4 w-4 rounded-full border border-primary/50 bg-background shadow transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50" />
    </SliderPrimitive.Root>
  )
})
Slider.displayName = "Slider"

export { Slider }
