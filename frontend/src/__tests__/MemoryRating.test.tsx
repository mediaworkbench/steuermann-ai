import { fireEvent, render, screen, waitFor } from "@testing-library/react";

import { MemoryRating } from "@/components/MemoryRating";
import { rateMemory } from "@/lib/api";

jest.mock("@/lib/api", () => ({
  rateMemory: jest.fn(),
}));

const mockRateMemory = rateMemory as jest.MockedFunction<typeof rateMemory>;

describe("MemoryRating", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockRateMemory.mockResolvedValue(true);
  });

  test("renders five stars", () => {
    render(<MemoryRating memoryId="mem-1" initialRating={3} />);

    expect(screen.getAllByRole("button")).toHaveLength(5);
  });

  test("submits selected rating", async () => {
    const onRatingChange = jest.fn();
    render(<MemoryRating memoryId="mem-1" initialRating={0} onRatingChange={onRatingChange} />);

    fireEvent.click(screen.getByLabelText("Rate 4 stars"));

    await waitFor(() => {
      expect(mockRateMemory).toHaveBeenCalledWith("mem-1", 4);
    });
    expect(onRatingChange).toHaveBeenCalledWith(4);
  });

  test("supports rating all stars from 1 to 5", async () => {
    render(<MemoryRating memoryId="mem-2" initialRating={0} />);

    for (let rating = 1; rating <= 5; rating += 1) {
      fireEvent.click(screen.getByLabelText(`Rate ${rating} star${rating === 1 ? "" : "s"}`));
      // eslint-disable-next-line no-await-in-loop
      await waitFor(() => {
        expect(mockRateMemory).toHaveBeenLastCalledWith("mem-2", rating);
      });
    }

    expect(mockRateMemory).toHaveBeenCalledTimes(5);
  });

  test("reverts UI callback when API call fails", async () => {
    mockRateMemory.mockResolvedValue(false);
    const onRatingChange = jest.fn();

    render(<MemoryRating memoryId="mem-3" initialRating={2} onRatingChange={onRatingChange} />);

    fireEvent.click(screen.getByLabelText("Rate 5 stars"));

    await waitFor(() => {
      expect(mockRateMemory).toHaveBeenCalledWith("mem-3", 5);
    });

    expect(onRatingChange).toHaveBeenNthCalledWith(1, 5);
    expect(onRatingChange).toHaveBeenNthCalledWith(2, 2);
  });
});
