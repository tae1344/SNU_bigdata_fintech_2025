import { NextResponse } from 'next/server';

// 미국 주별 불륜률 데이터 (CSV 데이터 기반)
const infidelityData = [
  { State: "Hawaii", Rank: 1, "Infidelity Rate": "80.56%" },
  { State: "Louisiana", Rank: 2, "Infidelity Rate": "62.50%" },
  { State: "Rhode Island", Rank: 2, "Infidelity Rate": "62.50%" },
  { State: "Delaware", Rank: 4, "Infidelity Rate": "52.94%" },
  { State: "Wyoming", Rank: 5, "Infidelity Rate": "50.00%" },
  { State: "New Hampshire", Rank: 5, "Infidelity Rate": "0.00%" },
  { State: "Maine", Rank: 7, "Infidelity Rate": "46.67%" },
  { State: "North Carolina", Rank: 8, "Infidelity Rate": "44.44%" },
  { State: "Utah", Rank: 10, "Infidelity Rate": "41.94%" },
  { State: "Nevada", Rank: 10, "Infidelity Rate": "41.94%" },
  { State: "Illinois", Rank: 10, "Infidelity Rate": "41.94%" },
  { State: "Tennessee", Rank: 13, "Infidelity Rate": "41.67%" },
  { State: "Vermont", Rank: 14, "Infidelity Rate": "40.00%" },
  { State: "Connecticut", Rank: 14, "Infidelity Rate": "40.00%" },
  { State: "Michigan", Rank: 16, "Infidelity Rate": "39.39%" },
  { State: "Arizona", Rank: 17, "Infidelity Rate": "38.89%" },
  { State: "New York", Rank: 18, "Infidelity Rate": "37.50%" },
  { State: "Kentucky", Rank: 20, "Infidelity Rate": "35.29%" },
  { State: "Alabama", Rank: 21, "Infidelity Rate": "34.38%" },
  { State: "West Virginia", Rank: 22, "Infidelity Rate": "33.33%" },
  { State: "Texas", Rank: 22, "Infidelity Rate": "33.33%" },
  { State: "Ohio", Rank: 22, "Infidelity Rate": "33.33%" },
  { State: "Wisconsin", Rank: 22, "Infidelity Rate": "33.33%" },
  { State: "Missouri", Rank: 22, "Infidelity Rate": "33.33%" },
  { State: "Georgia", Rank: 27, "Infidelity Rate": "32.35%" },
  { State: "Oklahoma", Rank: 28, "Infidelity Rate": "32.26%" },
  { State: "Pennsylvania", Rank: 29, "Infidelity Rate": "31.58%" },
  { State: "New Jersey", Rank: 30, "Infidelity Rate": "31.43%" },
  { State: "South Dakota", Rank: 31, "Infidelity Rate": "30.56%" },
  { State: "California", Rank: 32, "Infidelity Rate": "29.03%" },
  { State: "Indiana", Rank: 32, "Infidelity Rate": "29.03%" },
  { State: "Arkansas", Rank: 34, "Infidelity Rate": "28.57%" },
  { State: "Colorado", Rank: 35, "Infidelity Rate": "27.27%" },
  { State: "Minnesota", Rank: 35, "Infidelity Rate": "27.27%" },
  { State: "Alaska", Rank: 37, "Infidelity Rate": "26.47%" },
  { State: "Virginia", Rank: 37, "Infidelity Rate": "26.47%" },
  { State: "Idaho", Rank: 37, "Infidelity Rate": "26.47%" },
  { State: "Massachusetts", Rank: 40, "Infidelity Rate": "28.71%" },
  { State: "South Carolina", Rank: 41, "Infidelity Rate": "25.00%" },
  { State: "Iowa", Rank: 41, "Infidelity Rate": "25.00%" },
  { State: "Kansas", Rank: 43, "Infidelity Rate": "24.24%" },
  { State: "Maryland", Rank: 44, "Infidelity Rate": "23.33%" },
  { State: "Montana", Rank: 45, "Infidelity Rate": "17.65%" },
  { State: "Washington", Rank: 45, "Infidelity Rate": "17.65%" },
  { State: "Oregon", Rank: 47, "Infidelity Rate": "16.67%" },
  { State: "Mississippi", Rank: 48, "Infidelity Rate": "10.00%" },
  { State: "Florida", Rank: 49, "Infidelity Rate": "9.09%" },
  { State: "Nebraska", Rank: 50, "Infidelity Rate": "6.25%" }
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
