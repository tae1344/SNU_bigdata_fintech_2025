import { NextResponse } from 'next/server';

// 미국 주별 바람지수 데이터 (샘플 데이터)
const infidelityData = [
  { State: "Alabama", Rank: 1, "Infidelity Rate": "20.4%" },
  { State: "Alaska", Rank: 2, "Infidelity Rate": "19.8%" },
  { State: "Arizona", Rank: 3, "Infidelity Rate": "19.2%" },
  { State: "Arkansas", Rank: 4, "Infidelity Rate": "18.9%" },
  { State: "California", Rank: 5, "Infidelity Rate": "18.5%" },
  { State: "Colorado", Rank: 6, "Infidelity Rate": "18.1%" },
  { State: "Connecticut", Rank: 7, "Infidelity Rate": "17.8%" },
  { State: "Delaware", Rank: 8, "Infidelity Rate": "17.5%" },
  { State: "Florida", Rank: 9, "Infidelity Rate": "17.2%" },
  { State: "Georgia", Rank: 10, "Infidelity Rate": "16.9%" },
  { State: "Hawaii", Rank: 11, "Infidelity Rate": "16.6%" },
  { State: "Idaho", Rank: 12, "Infidelity Rate": "16.3%" },
  { State: "Illinois", Rank: 13, "Infidelity Rate": "16.0%" },
  { State: "Indiana", Rank: 14, "Infidelity Rate": "15.7%" },
  { State: "Iowa", Rank: 15, "Infidelity Rate": "15.4%" },
  { State: "Kansas", Rank: 16, "Infidelity Rate": "15.1%" },
  { State: "Kentucky", Rank: 17, "Infidelity Rate": "14.8%" },
  { State: "Louisiana", Rank: 18, "Infidelity Rate": "14.5%" },
  { State: "Maine", Rank: 19, "Infidelity Rate": "14.2%" },
  { State: "Maryland", Rank: 20, "Infidelity Rate": "13.9%" },
  { State: "Massachusetts", Rank: 21, "Infidelity Rate": "13.6%" },
  { State: "Michigan", Rank: 22, "Infidelity Rate": "13.3%" },
  { State: "Minnesota", Rank: 23, "Infidelity Rate": "13.0%" },
  { State: "Mississippi", Rank: 24, "Infidelity Rate": "12.7%" },
  { State: "Missouri", Rank: 25, "Infidelity Rate": "12.4%" },
  { State: "Montana", Rank: 26, "Infidelity Rate": "12.1%" },
  { State: "Nebraska", Rank: 27, "Infidelity Rate": "11.8%" },
  { State: "Nevada", Rank: 28, "Infidelity Rate": "11.5%" },
  { State: "New Hampshire", Rank: 29, "Infidelity Rate": "11.2%" },
  { State: "New Jersey", Rank: 30, "Infidelity Rate": "10.9%" },
  { State: "New Mexico", Rank: 31, "Infidelity Rate": "10.6%" },
  { State: "New York", Rank: 32, "Infidelity Rate": "10.3%" },
  { State: "North Carolina", Rank: 33, "Infidelity Rate": "10.0%" },
  { State: "North Dakota", Rank: 34, "Infidelity Rate": "9.7%" },
  { State: "Ohio", Rank: 35, "Infidelity Rate": "9.4%" },
  { State: "Oklahoma", Rank: 36, "Infidelity Rate": "9.1%" },
  { State: "Oregon", Rank: 37, "Infidelity Rate": "8.8%" },
  { State: "Pennsylvania", Rank: 38, "Infidelity Rate": "8.5%" },
  { State: "Rhode Island", Rank: 39, "Infidelity Rate": "8.2%" },
  { State: "South Carolina", Rank: 40, "Infidelity Rate": "7.9%" },
  { State: "South Dakota", Rank: 41, "Infidelity Rate": "7.6%" },
  { State: "Tennessee", Rank: 42, "Infidelity Rate": "7.3%" },
  { State: "Texas", Rank: 43, "Infidelity Rate": "7.0%" },
  { State: "Utah", Rank: 44, "Infidelity Rate": "6.7%" },
  { State: "Vermont", Rank: 45, "Infidelity Rate": "6.4%" },
  { State: "Virginia", Rank: 46, "Infidelity Rate": "6.1%" },
  { State: "Washington", Rank: 47, "Infidelity Rate": "5.8%" },
  { State: "West Virginia", Rank: 48, "Infidelity Rate": "5.5%" },
  { State: "Wisconsin", Rank: 49, "Infidelity Rate": "5.2%" },
  { State: "Wyoming", Rank: 50, "Infidelity Rate": "4.9%" }
];

export async function GET() {
  try {
    return NextResponse.json(infidelityData);
  } catch (error) {
    return NextResponse.json(
      { error: '데이터를 불러오는데 실패했습니다.' },
      { status: 500 }
    );
  }
}
