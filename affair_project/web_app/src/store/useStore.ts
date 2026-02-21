import { AppActions, AppState, Filters } from "@/types";
import { create } from "zustand";
import { devtools } from "zustand/middleware";

const DEFAULT_FILTERS: Filters = {
  rate_marriage: 3,
  age: 35,
  yrs_married: 8,
  children: 1,
  religious: 2,
  educ: 3,
  occupation: 2,
  occupation_husb: 2,
};

export const useStore = create<AppState & AppActions>()(
  devtools(
    (set, get) => ({
      // Initial state
      currentStep: "landing",
      filters: DEFAULT_FILTERS,
      result: null,
      previousResult: null,
      loading: true,

      // Actions
      setCurrentStep: (step) => {
        set({ currentStep: step });
      },

      setLoading: (loading: boolean) => {
        set({ loading });
      },

      updateFilter: (key, value) =>
        set((state) => ({
          filters: { ...state.filters, [key]: value },
        })),
      
      setResult: (result) => set({ result }),
      
      setPreviousResult: (result) => set({ previousResult: result }),
      
      resetToLanding: () => set({
        currentStep: "landing",
        filters: DEFAULT_FILTERS,
        result: null,
        previousResult: null,
        loading: false, // 여기도 false로 변경
      }),
    }),
    { name: "app-store" }
  )
);
