import { Card, CardContent } from "@/components/ui/card";
import { COLORS } from "../styles/Colors";

export default function StatCard({ icon: Icon, label, value, color = COLORS.primary }: { 
  icon: React.ComponentType<{ size?: number }>; 
  label: string; 
  value: string;
  color?: string;
}) {
  return (
    <Card 
      className="border-0 shadow-lg" 
      style={{ backgroundColor: COLORS.dark.background.secondary }}
      role="region"
      aria-label={`${label}: ${value}`}
    >
      <CardContent className="p-4 flex items-center gap-3">
        <div 
          className="p-2 rounded-xl" 
          style={{ backgroundColor: color, color: 'white' }}
          aria-hidden="true"
        >
          <Icon size={20} />
        </div>
        <div>
          <div className="text-sm font-medium" style={{ color: COLORS.white }}>{label}</div>
          <div className="text-lg font-semibold" style={{ color: COLORS.white }}>{value}</div>
        </div>
      </CardContent>
    </Card>
  );
}