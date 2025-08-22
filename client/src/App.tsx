import { Flex, Select, SimpleGrid, Slider, Stack, Text, Switch } from "@mantine/core";
import React from "react";
import P5Wrapper from 'react-p5-wrapper';
import GyroVisualizer from "./GyroVisualizer";

export type DataFile = {
    name: string,
    url: string,
    size: number,
    modified: number,
}

export default function App() {
  const [files, setFiles] = React.useState<DataFile[]>([]);

  const [fieldScale, setFieldScale] = React.useState<number>(2);
  const [warp, setWarp] = React.useState<number>(0.4);
  const [octaves, setOctaves] = React.useState<number>(2);
  const [speed, setSpeed] = React.useState<number>(1);
  const [secondShader, setSecondShader] = React.useState<boolean>(false);

  const [componentConfig, setComponentConfig] = React.useState<{
    x: "x",
    y: "y",
    z: "z",
  }>({x:"x",y:"y",z:"z"});

  const [selectedFile, setSelectedFile] = React.useState<DataFile>();

  const fetchFiles = async () => {
    return (await (await fetch("/api/files")).json()).files;
  }

  React.useEffect(() => {
    fetchFiles().then(f => setFiles(f));
  }, []);

  return (
    <Stack justify="center" align="center">
      <Select
        label="Data Source"
        data={files.map(f => f.name)}
        w={500}
        value={selectedFile?.name}
        onChange={f => setSelectedFile(files.find(e => e.name == f))}
      />
      <SimpleGrid cols={2} w="40%">
        <Text>Field Scale: </Text>
        <Slider size="md" defaultValue={fieldScale} onChange={setFieldScale} min={0.1} step={0.1} max={10} />
        <Text>Warp: </Text>
        <Slider disabled={secondShader} size="md" defaultValue={warp} onChange={setWarp} min={0.1} step={0.1} max={5} />
        <Text>Sharpness (Very laggy &gt;1): </Text>
        <Slider disabled={secondShader} size="md" defaultValue={octaves} onChange={setOctaves} min={1} step={1} max={4}/>
        <Text>Speed: </Text>
        <Slider size="md" defaultValue={speed} onChange={setSpeed} min={0.1} max={5} />
        <Text>Blockier</Text>
        <Switch checked={secondShader} onChange={e => setSecondShader(e.currentTarget.checked)} />
      </SimpleGrid>
      {selectedFile && <GyroVisualizer
        csvUrl={selectedFile.url}
        width={1280}
        height={720}
        loopDurationMs={15000}
        palette="terrain"     // or "grayscale"
        smoothingMs={700}
        blendWindowMs={300}
        fieldScale={fieldScale}
        warp={warp}
        octaves={octaves}
        shaderOption={secondShader ? 2 : 1}
        componentConfig={componentConfig}
        speed={speed}
      />}
    </Stack>
  );
}