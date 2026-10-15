import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parents[3])  # repo root; paths below are relative to it

import json

with open("all_250_evaluated.json") as f:
    data = json.load(f)

# Compact data for canvas
compact_data = []
for d in data:
    compact_data.append({
        "id": d["id"],
        "num": d["num"],
        "set": d["set_name"],
        "file": d["file"],
        "cat": d["category"],
        "title": d["title"],
        "idea": d["core_idea"][:160] + ("..." if len(d["core_idea"]) > 160 else ""),
        "exp": d["experiment"][:160] + ("..." if len(d["experiment"]) > 160 else ""),
        "kill": d["kill_test"][:120] if d.get("kill_test") else "",
        "comp": d["compute_note"],
        "nips": d["neurips"],
        "icml": d["icml"],
        "iclr": d["iclr"],
        "aaai": d["aaai"],
        "feas": d["feasibility_3080"],
        "avg": d["avg_conf"],
        "sweet": d["sweet_spot"],
        "best": d["best_conf"],
        "c_why": d["conf_reason"],
        "m_why": d["comp_reason"]
    })

data_json = json.dumps(compact_data)

ts_template = """import {
  useState,
  useMemo,
  useHostTheme,
  Stack,
  Row,
  Grid,
  Card,
  CardHeader,
  CardBody,
  Stat,
  Table,
  Button,
  TextInput,
  Select,
  Pill,
  Text,
  H1,
  H2,
  Divider,
  Spacer
} from "cursor/canvas";

interface Idea {
  id: string;
  num: number;
  set: string;
  file: string;
  cat: string;
  title: string;
  idea: string;
  exp: string;
  kill: string;
  comp: string;
  nips: number;
  icml: number;
  iclr: number;
  aaai: number;
  feas: number;
  avg: number;
  sweet: number;
  best: string;
  c_why: string;
  m_why: string;
}

const ALL_IDEAS: Idea[] = %DATA_JSON%;

export default function RLResearchRankings() {
  const theme = useHostTheme();
  const [search, setSearch] = useState("");
  const [setFilter, setSetFilter] = useState("all");
  const [catFilter, setCatFilter] = useState("all");
  const [sortBy, setSortBy] = useState<"sweet" | "feas" | "nips" | "icml" | "iclr" | "aaai">("sweet");
  const [selectedId, setSelectedId] = useState<string>(ALL_IDEAS[0].id);

  const filteredIdeas = useMemo(() => {
    return ALL_IDEAS.filter((item) => {
      if (setFilter !== "all" && item.file !== setFilter) return false;
      if (catFilter !== "all" && item.cat !== catFilter) return false;
      if (search.trim() !== "") {
        const q = search.toLowerCase();
        const match =
          item.title.toLowerCase().includes(q) ||
          item.idea.toLowerCase().includes(q) ||
          item.cat.toLowerCase().includes(q) ||
          item.set.toLowerCase().includes(q);
        if (!match) return false;
      }
      return true;
    }).sort((a, b) => {
      if (sortBy === "sweet") return b.sweet - a.sweet;
      if (sortBy === "feas") return b.feas - a.feas;
      if (sortBy === "nips") return b.nips - a.nips;
      if (sortBy === "icml") return b.icml - a.icml;
      if (sortBy === "iclr") return b.iclr - a.iclr;
      if (sortBy === "aaai") return b.aaai - a.aaai;
      return 0;
    });
  }, [search, setFilter, catFilter, sortBy]);

  const selectedIdea = useMemo(() => {
    return ALL_IDEAS.find((i) => i.id === selectedId) || ALL_IDEAS[0];
  }, [selectedId]);

  const stats = useMemo(() => {
    const sweetGems = ALL_IDEAS.filter((i) => i.sweet >= 84).length;
    const highFeas = ALL_IDEAS.filter((i) => i.feas >= 90).length;
    const topNips = ALL_IDEAS.filter((i) => i.nips >= 85).length;
    const topIcml = ALL_IDEAS.filter((i) => i.icml >= 88).length;
    return { total: ALL_IDEAS.length, sweetGems, highFeas, topNips, topIcml };
  }, []);

  const tableHeaders = [
    "#",
    "Title & Set",
    "Category",
    "NeurIPS",
    "ICML",
    "ICLR",
    "AAAI",
    "3080 Feas",
    "Sweet Spot",
    "Action"
  ];

  const tableRows = filteredIdeas.slice(0, 60).map((idea, idx) => [
    <Text key="idx" weight="medium">{idx + 1}</Text>,
    <Stack key="title" gap={2}>
      <Text weight="medium">{idea.title}</Text>
      <Text size="small" tone="secondary">{idea.set} • #{idea.num}</Text>
    </Stack>,
    <Pill key="cat" size="sm">{idea.cat}</Pill>,
    <Text key="nips" weight="medium">{idea.nips}</Text>,
    <Text key="icml" weight="medium">{idea.icml}</Text>,
    <Text key="iclr" weight="medium">{idea.iclr}</Text>,
    <Text key="aaai" weight="medium">{idea.aaai}</Text>,
    <Text key="feas" weight="semibold">{idea.feas}</Text>,
    <Pill key="sweet" size="sm" active={idea.sweet >= 85}>{idea.sweet}</Pill>,
    <Button
      key="btn"
      variant={selectedId === idea.id ? "primary" : "ghost"}
      onClick={() => setSelectedId(idea.id)}
    >
      Inspect
    </Button>
  ]);

  return (
    <Stack gap={20} style={{ padding: 24, background: theme.bg.editor }}>
      <Row align="center" justify="space-between">
        <Stack gap={4}>
          <H1>RL Research Ideas: Top-4 Conference & RTX 3080 Rankings</H1>
          <Text tone="secondary">
            Comprehensive evaluation of 250 RL research ideas across NeurIPS, ICML, ICLR, AAAI and RTX 3080 single-GPU feasibility.
          </Text>
        </Stack>
      </Row>

      <Grid columns={4} gap={12}>
        <Card variant="borderless">
          <CardBody>
            <Stat label="Total Ideas Analyzed" value={stats.total} />
          </CardBody>
        </Card>
        <Card variant="borderless">
          <CardBody>
            <Stat
              label="Sweet Spot Contenders (Score ≥ 84)"
              value={stats.sweetGems}
              tone="success"
            />
          </CardBody>
        </Card>
        <Card variant="borderless">
          <CardBody>
            <Stat
              label="Ultra-Fast 3080 Feasible (Score ≥ 90)"
              value={stats.highFeas}
              tone="info"
            />
          </CardBody>
        </Card>
        <Card variant="borderless">
          <CardBody>
            <Stat
              label="Elite Theory & Algorithmic (Score ≥ 88)"
              value={stats.topIcml}
            />
          </CardBody>
        </Card>
      </Grid>

      {selectedIdea && (
        <Card variant="default">
          <CardHeader trailing={<Pill size="sm" active>Sweet Spot: {selectedIdea.sweet}</Pill>}>
            {`Idea #${selectedIdea.num}: ${selectedIdea.title}`}
          </CardHeader>
          <CardBody>
            <Grid columns="2fr 1fr" gap={16}>
              <Stack gap={12}>
                <Row gap={8}>
                  <Pill size="sm">{selectedIdea.set}</Pill>
                  <Pill size="sm">{selectedIdea.cat}</Pill>
                  <Pill size="sm" active>Best Fit: {selectedIdea.best}</Pill>
                </Row>
                <div>
                  <Text weight="semibold">Core Hypothesis / Question:</Text>
                  <Text tone="secondary">{selectedIdea.idea}</Text>
                </div>
                <div>
                  <Text weight="semibold">Experimental Protocol:</Text>
                  <Text tone="secondary">{selectedIdea.exp}</Text>
                </div>
                {selectedIdea.kill && (
                  <div>
                    <Text weight="semibold">Kill Test:</Text>
                    <Text tone="secondary">{selectedIdea.kill}</Text>
                  </div>
                )}
              </Stack>

              <Stack gap={10} style={{ borderLeft: `1px solid ${theme.stroke.tertiary}`, paddingLeft: 16 }}>
                <Text weight="semibold">Scores Breakdown (1-100):</Text>
                <Row justify="space-between">
                  <Text tone="secondary">NeurIPS Acceptance:</Text>
                  <Text weight="medium">{selectedIdea.nips}</Text>
                </Row>
                <Row justify="space-between">
                  <Text tone="secondary">ICML Acceptance:</Text>
                  <Text weight="medium">{selectedIdea.icml}</Text>
                </Row>
                <Row justify="space-between">
                  <Text tone="secondary">ICLR Acceptance:</Text>
                  <Text weight="medium">{selectedIdea.iclr}</Text>
                </Row>
                <Row justify="space-between">
                  <Text tone="secondary">AAAI Acceptance:</Text>
                  <Text weight="medium">{selectedIdea.aaai}</Text>
                </Row>
                <Divider />
                <Row justify="space-between">
                  <Text weight="semibold">RTX 3080 Feasibility:</Text>
                  <Text weight="bold">{selectedIdea.feas}</Text>
                </Row>
                <Text size="small" tone="secondary">
                  <strong>Compute Note:</strong> {selectedIdea.comp}
                </Text>
                <Text size="small" tone="secondary">
                  <strong>Analysis:</strong> {selectedIdea.m_why}
                </Text>
              </Stack>
            </Grid>
          </CardBody>
        </Card>
      )}

      <Stack gap={12}>
        <H2>All 250 Ideas Ranked</H2>
        <Row gap={12} align="center" wrap>
          <TextInput
            value={search}
            onChange={setSearch}
            placeholder="Search ideas, keywords, environments..."
            style={{ minWidth: 260 }}
          />
          <Select
            value={setFilter}
            onChange={setSetFilter}
            options={[
              { value: "all", label: "All 5 Lists (250 Ideas)" },
              { value: "rl_ideas_50.md", label: "Set 1: Paper Follow-ups (50)" },
              { value: "rl_ideas_novel_50.md", label: "Novel Set 1: Mechanisms (50)" },
              { value: "rl_ideas_novel_50_set2.md", label: "Novel Set 2: Hyperparams (50)" },
              { value: "rl_ideas_novel_50_set3.md", label: "Novel Set 3: Dynamics (50)" },
              { value: "rl_ideas_novel_50_set4.md", label: "Novel Set 4: Value/Control (50)" }
            ]}
          />
          <Select
            value={catFilter}
            onChange={setCatFilter}
            options={[
              { value: "all", label: "All Categories" },
              { value: "Learning Dynamics & Optimization", label: "Learning Dynamics & Optimization" },
              { value: "Architecture & Scaling", label: "Architecture & Scaling" },
              { value: "Exploration", label: "Exploration" },
              { value: "World Models & MBRL", label: "World Models & MBRL" },
              { value: "Offline RL & Data", label: "Offline RL & Data" },
              { value: "Control, Robotics & Safety", label: "Control, Robotics & Safety" },
              { value: "LLM RL & Reasoning", label: "LLM RL & Reasoning" },
              { value: "Multi-Agent RL", label: "Multi-Agent RL" },
              { value: "Evaluation & Methodology", label: "Evaluation & Methodology" }
            ]}
          />
          <Spacer />
          <Select
            value={sortBy}
            onChange={(v) => setSortBy(v as any)}
            options={[
              { value: "sweet", label: "Sort: Sweet Spot (Best Overall)" },
              { value: "feas", label: "Sort: RTX 3080 Feasibility" },
              { value: "nips", label: "Sort: NeurIPS Likelihood" },
              { value: "icml", label: "Sort: ICML Likelihood" },
              { value: "iclr", label: "Sort: ICLR Likelihood" },
              { value: "aaai", label: "Sort: AAAI Likelihood" }
            ]}
          />
        </Row>

        <Text size="small" tone="secondary">
          Showing {Math.min(60, filteredIdeas.length)} of {filteredIdeas.length} matching ideas. Click "Inspect" to view full experimental design in the card above.
        </Text>

        <Table headers={tableHeaders} rows={tableRows} striped />
      </Stack>
    </Stack>
  );
}
"""

final_code = ts_template.replace("%DATA_JSON%", data_json)

canvas_path = "/Users/nathanclark/.cursor/projects/Users-nathanclark-dev-havn/canvases/rl-research-rankings.canvas.tsx"
with open(canvas_path, "w") as f:
    f.write(final_code)

print("Canvas successfully written to", canvas_path)
