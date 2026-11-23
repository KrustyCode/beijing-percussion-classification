// constants/labels.ts
export type ClassId = "bangu" | "daluo" | "naobo" | "xiaoluo";

export interface ClassInfo {
  id: ClassId;
  name: string; // display name
  nameZh: string; // Chinese name
  description: string; // short English description
}

export const CLASS_BY_ID: Record<ClassId, ClassInfo> = {
  bangu: {
    id: "bangu",
    name: "Bangu",
    nameZh: "板鼓",
    description: "Small drum",
  },
  daluo: {
    id: "daluo",
    name: "Daluo",
    nameZh: "大锣",
    description: "Large gong",
  },
  naobo: {
    id: "naobo",
    name: "Naobo",
    nameZh: "铙钹",
    description: "Cymbals",
  },
  xiaoluo: {
    id: "xiaoluo",
    name: "Xiaoluo",
    nameZh: "小锣",
    description: "Small gong",
  },
};

export const CLASS_ORDER: readonly ClassId[] = [
  "bangu",
  "daluo",
  "naobo",
  "xiaoluo",
];
export const CLASSES: readonly ClassInfo[] = CLASS_ORDER.map(
  (id) => CLASS_BY_ID[id],
);

export function isClassId(value: unknown): value is ClassId {
  return typeof value === "string" && value in CLASS_BY_ID;
}
