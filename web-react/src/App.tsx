import React, { useState, useCallback, useEffect, useRef } from 'react';
import { FileUploadComponent } from './components/FileUpload';
import { QuestionExplorer } from './components/QuestionExplorer';
import { StatsDashboard } from './components/StatsDashboard';
import { EnhancedResultItem, ConversationIndexEntry, ConversationMessage, EpisodeRef, SemanticMemoryRef } from './types';
import './App.css';

// 类别映射（忽略类别5）
const CATEGORY_LABEL_MAP: Record<string, string> = {
  '1': '多跳',
  '2': '时间',
  '3': '开放性',
  '4': '单跳'
};

// 工具：标准化文本用于匹配
const normalizeText = (text: string) => text.toLowerCase().trim();

function App() {
  const [enhancedData, setEnhancedData] = useState<EnhancedResultItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [initializedTrace, setInitializedTrace] = useState(false);

  // 索引：对话、用户键、记忆
  const conversationIndexRef = useRef<Map<number, ConversationIndexEntry>>(new Map());
  const convIdxToUserKeyRef = useRef<Map<number, string>>(new Map());
  const episodeTextIndexRef = useRef<Map<string, Map<string, { title?: string }>>>(new Map()); // user_key -> text -> {episode_id}
  const episodeIdToTitleRef = useRef<Map<string, string>>(new Map()); // global episode_id -> title
  const semanticByEpisodeRef = useRef<Map<string, SemanticMemoryRef[]>>(new Map()); // global episode_id -> semantic list

  const applyCategoryMapping = useCallback((items: EnhancedResultItem[]) => {
    return items
      .filter(item => Number(item.category) !== 5)
      .map(item => {
        const numericCategory = Number(item.category);
        const mapped = CATEGORY_LABEL_MAP[String(numericCategory)] ?? String(item.category);
        return { ...item, category: mapped };
      });
  }, []);

  const handleDataProcessed = useCallback((data: EnhancedResultItem[]) => {
    // 1) 类别映射
    const transformed = applyCategoryMapping(data);
    // 2) 若上传时未提供数据集证据，尝试根据 results.evidence 指针与内置对话索引补全原文
    const enriched = transformed.map((item) => {
      const convIdx = Number(item.conversation_id ?? (item as any).conversationId);
      const conv = conversationIndexRef.current.get(convIdx);
      const pointers = parseEvidencePointers((item as any).evidence || []);
      const originals: ConversationMessage[] = [];
      if (conv && pointers.length > 0) {
        pointers.forEach((p) => {
          const msg = conv.diaIdToMessage[p];
          if (msg) originals.push(msg);
        });
      }
      return {
        ...item,
        evidence_pointers: pointers.length > 0 ? pointers : item.evidence_pointers,
        evidence_original_messages: originals.length > 0 ? originals : item.evidence_original_messages
      } as EnhancedResultItem;
    });

    setEnhancedData(enriched);
    setError(null);
  }, [applyCategoryMapping]);

  const handleError = useCallback((errorMessage: string) => {
    setError(errorMessage);
    setEnhancedData([]);
  }, []);

  const handleLoadingChange = useCallback((loading: boolean) => {
    setIsLoading(loading);
  }, []);

  // 解析对话，构建 dia_id -> message 索引
  const buildConversationIndex = (conversation: any, speakerA: string, speakerB: string): ConversationIndexEntry => {
    const diaIdToMessage: Record<string, ConversationMessage> = {};
    Object.keys(conversation || {}).forEach((key) => {
      if (/^session_\d+$/.test(key) && Array.isArray(conversation[key])) {
        (conversation[key] as any[]).forEach((m: any) => {
          if (m && typeof m.dia_id === 'string') {
            diaIdToMessage[m.dia_id] = {
              speaker: String(m.speaker ?? ''),
              dia_id: String(m.dia_id),
              text: String(m.text ?? ''),
              img_url: Array.isArray(m.img_url) ? m.img_url : undefined,
              blip_caption: m.blip_caption,
              query: m.query
            };
          }
        });
      }
    });
    return {
      user_key: '', // 稍后外层赋值
      speakers: { a: speakerA, b: speakerB },
      diaIdToMessage
    };
  };

  // 解析证据指针：将 ['D1:3', 'D8:6; D9:17'] -> ['D1:3','D8:6','D9:17']
  const parseEvidencePointers = (evidence: any): string[] => {
    const tokens: string[] = [];
    if (Array.isArray(evidence)) {
      evidence.forEach((entry) => {
        if (typeof entry === 'string') {
          entry.split(/[;，,]+/).forEach((seg) => {
            const t = seg.trim();
            if (/^D\d+:\d+$/.test(t)) tokens.push(t);
          });
        }
      });
    }
    return tokens;
  };

  // 启动时自动从 /public/locomo10.json 加载基础数据，并构建对话索引与证据原文
  useEffect(() => {
    const loadBuiltinDataset = async () => {
      setIsLoading(true);
      try {
        const res = await fetch('/locomo10.json');
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        const raw = await res.json();
        // 期望结构：Array<{ qa: Array<{ question, answer, evidence, category, adversarial_answer? }> }>
        const enhanced: EnhancedResultItem[] = [];
        if (Array.isArray(raw)) {
          raw.forEach((item: any, convIdx: number) => {
            const qaList = Array.isArray(item?.qa) ? item.qa : [];
            const conversation = item?.conversation || {};
            const speakerA = String(conversation?.speaker_a ?? '');
            const speakerB = String(conversation?.speaker_b ?? '');

            // user_key 推断：按文件顺序与 memories 命名规则一致，例如 Caroline_0、Jon_1...
            const userKey = `${speakerA}_${convIdx}`;
            const convEntry = buildConversationIndex(conversation, speakerA, speakerB);
            convEntry.user_key = userKey;
            conversationIndexRef.current.set(convIdx, convEntry);
            convIdxToUserKeyRef.current.set(convIdx, userKey);

            qaList.forEach((qa: any) => {
              const categoryNum = Number(qa?.category);
              if (categoryNum === 5) return; // 忽略第五类
              const mappedCategory = CATEGORY_LABEL_MAP[String(categoryNum)] ?? String(qa?.category ?? '');
              const pointers = parseEvidencePointers(qa?.evidence);
              const originals: ConversationMessage[] = [];
              pointers.forEach((p) => {
                const msg = convEntry.diaIdToMessage[p];
                if (msg) originals.push(msg);
              });
              enhanced.push({
                question: String(qa?.question ?? ''),
                answer: String(qa?.answer ?? ''),
                category: mappedCategory,
                evidence: Array.isArray(qa?.evidence) ? qa.evidence : [],
                response: '',
                adversarial_answer: qa?.adversarial_answer,
                speaker_1_memories: [],
                speaker_2_memories: [],
                num_speaker_1_memories: 0,
                num_speaker_2_memories: 0,
                speaker_1_memory_time: 0,
                speaker_2_memory_time: 0,
                response_time: 0,
                evidence_text: Array.isArray(qa?.evidence) ? qa.evidence : [],
                conversation_id: String(convIdx),
                is_correct: undefined,
                has_evidence: Array.isArray(qa?.evidence) ? qa.evidence.length > 0 : false,
                memory_count: 0,
                evidence_pointers: pointers,
                evidence_original_messages: originals
              });
            });
          });
        }
        setEnhancedData(enhanced);
        setError(null);
      } catch (e) {
        const message = e instanceof Error ? e.message : String(e);
        setError(`内置数据加载失败：${message}`);
      } finally {
        setIsLoading(false);
      }
    };
    loadBuiltinDataset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 构建 memories 索引：episodes 与 semantic（一次性加载）
  useEffect(() => {
    const buildMemoriesIndex = async () => {
      if (initializedTrace) return;
      try {
        const idxRes = await fetch('/memories/index.json');
        if (!idxRes.ok) return; // 若不存在则跳过
        const { episodes = [], semantic = [] } = await idxRes.json();

        // 逐用户加载 episodes
        for (const epFile of episodes as string[]) {
          // 文件名形如 Caroline_0_episodes.jsonl
          const userKey = epFile.replace(/_episodes\.jsonl$/, '');
          const url = `/memories/episodes/${epFile}`;
          const text = await (await fetch(url)).text();
          const lines = text.split(/\n+/).map(l => l.trim()).filter(Boolean);
          let userTextMap = episodeTextIndexRef.current.get(userKey);
          if (!userTextMap) {
            userTextMap = new Map();
            episodeTextIndexRef.current.set(userKey, userTextMap);
          }
          for (const line of lines) {
            try {
              const obj = JSON.parse(line);
              const episodeId: string = obj.episode_id;
              const title: string | undefined = obj.title;
              if (episodeId) {
                if (title) episodeIdToTitleRef.current.set(episodeId, title);
                const originalMsgs: any[] = Array.isArray(obj.original_messages) ? obj.original_messages : [];
                for (const m of originalMsgs) {
                  const content = typeof m.content === 'string' ? normalizeText(m.content) : '';
                  if (!content) continue;
                  const existing = userTextMap.get(content) || {};
                  userTextMap.set(content, { ...(existing as any), [episodeId]: { title } });
                }
              }
            } catch { /* 忽略坏行 */ }
          }
        }

        // 逐用户加载 semantic
        for (const smFile of semantic as string[]) {
          const url = `/memories/semantic/${smFile}`;
          const text = await (await fetch(url)).text();
          const lines = text.split(/\n+/).map(l => l.trim()).filter(Boolean);
          for (const line of lines) {
            try {
              const obj = JSON.parse(line);
              const memId: string = obj.memory_id;
              const content: string = obj.content;
              const confidence: number | undefined = obj.confidence;
              const src: string[] = Array.isArray(obj.source_episodes) ? obj.source_episodes : [];
              const ref: SemanticMemoryRef = { memory_id: memId, content, confidence };
              for (const epId of src) {
                const arr = semanticByEpisodeRef.current.get(epId) || [];
                arr.push(ref);
                semanticByEpisodeRef.current.set(epId, arr);
              }
            } catch { /* 忽略坏行 */ }
          }
        }

        setInitializedTrace(true);
      } catch {
        // 可忽略
      }
    };
    buildMemoriesIndex();
  }, [initializedTrace]);

  // 一旦 memories 索引可用，则为每道题链接情景记忆与语义记忆
  useEffect(() => {
    if (!initializedTrace || enhancedData.length === 0) return;
    // 若已存在链接信息则不重复计算
    const needLink = enhancedData.some((i) => !i.linked_episodes || !i.linked_semantic_memories);
    if (!needLink) return;
    const next = enhancedData.map((item) => {
      const convIdx = Number(item.conversation_id);
      const userKey = convIdxToUserKeyRef.current.get(convIdx);
      const userTextMap = userKey ? episodeTextIndexRef.current.get(userKey) : undefined;
      const linkedEpisodeIds = new Set<string>();
      const linkedEpisodes: EpisodeRef[] = [];
      const linkedSemantic: SemanticMemoryRef[] = [];

      (item.evidence_original_messages || []).forEach((m) => {
        const norm = normalizeText(m.text);
        const hits = userTextMap?.get(norm);
        if (hits) {
          Object.keys(hits).forEach((epId) => {
            if (!linkedEpisodeIds.has(epId)) {
              linkedEpisodeIds.add(epId);
              linkedEpisodes.push({ episode_id: epId, title: episodeIdToTitleRef.current.get(epId) });
              const sem = semanticByEpisodeRef.current.get(epId) || [];
              sem.forEach((s) => linkedSemantic.push(s));
            }
          });
        }
      });

      return {
        ...item,
        linked_episodes: linkedEpisodes,
        linked_semantic_memories: linkedSemantic
      } as EnhancedResultItem;
    });
    setEnhancedData(next);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initializedTrace, enhancedData]);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                LoCoMo 结果分析器
              </h1>
              <p className="text-sm text-gray-600 mt-1">
                智能记忆检索评测工具 - 支持结果对比与证据分析
              </p>
            </div>
            {enhancedData.length > 0 && (
              <div className="text-right">
                <div className="text-2xl font-bold text-primary-600">
                  {enhancedData.length}
                </div>
                <div className="text-sm text-gray-500">道题目已加载</div>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex">
              <div className="text-red-600">
                <h3 className="font-medium">数据处理错误</h3>
                <p className="text-sm mt-1">{error}</p>
              </div>
            </div>
          </div>
        )}

        {isLoading && (
          <div className="mb-6 p-8 text-center">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
            <p className="mt-2 text-gray-600">正在处理数据...</p>
          </div>
        )}

        {enhancedData.length === 0 && !isLoading ? (
          <div className="text-center py-12">
            <FileUploadComponent 
              onDataProcessed={handleDataProcessed}
              onError={handleError}
              onLoadingChange={handleLoadingChange}
            />
          </div>
        ) : (
          <div className="space-y-6">
            {/* Statistics Dashboard */}
            <StatsDashboard data={enhancedData} />
            
            {/* Question Explorer */}
            <QuestionExplorer data={enhancedData} />

            {/* Reset Button */}
            <div className="text-center pt-4">
              <button
                onClick={() => {
                  setEnhancedData([]);
                  setError(null);
                }}
                className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
              >
                重新上传数据
              </button>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-sm text-gray-500">
            © 2024 LoCoMo 结果分析器 - Nemori Memory System 评测工具
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
