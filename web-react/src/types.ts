// 数据类型定义

export interface Memory {
  memory: string;
  timestamp: string;
  score: number;
  episode_title?: string;
  episode_id: string;
  level: number;
  memory_type: 'episodic' | 'semantic';
  search_method: string;
  has_original_messages?: boolean;
  original_messages?: OriginalMessage[];
  knowledge_type?: string;
  confidence?: number;
  related_episodes?: string[];
}

export interface OriginalMessage {
  role: string;
  content: string;
  timestamp: string;
  speaker?: string;
}

export interface ResultItem {
  question: string;
  answer: string;
  category: string;
  evidence: string[];
  response: string;
  adversarial_answer?: string;
  speaker_1_memories: Memory[];
  speaker_2_memories: Memory[];
  num_speaker_1_memories: number;
  num_speaker_2_memories: number;
  speaker_1_memory_time: number;
  speaker_2_memory_time: number;
  response_time: number;
  system_version?: string;
}

export interface MetricItem {
  question: string;
  answer: string;
  response: string;
  category: string;
  bleu_score: number;
  f1_score: number;
  llm_score: number;
}

export interface DatasetEvidence {
  question: string;
  answer: string;
  category: string;
  evidence: string[];
  adversarial_answer?: string;
}

export interface DatasetQA {
  qa: DatasetEvidence[];
}

export interface DatasetItem {
  conversation: {
    speaker_a: string;
    speaker_b: string;
    [key: string]: any;
  };
  qa: DatasetEvidence[];
}

// 合并后的完整数据项
export interface EnhancedResultItem extends ResultItem {
  // 从metrics添加的字段
  bleu_score?: number;
  f1_score?: number;
  llm_score?: number;
  
  // 从数据集添加的字段
  evidence_text?: string[];
  conversation_id?: string;
  
  // 计算得出的字段
  is_correct?: boolean;
  has_evidence?: boolean;
  memory_count?: number;

  // 证据溯源新增字段
  evidence_pointers?: string[]; // 如 ["D1:3", "D1:5"]，将分号分割统一展平
  evidence_original_messages?: ConversationMessage[]; // 从对话中按 evidence_pointers 抽取的原文
  linked_episodes?: EpisodeRef[]; // 关联到的情景记忆（按证据汇总去重）
  linked_semantic_memories?: SemanticMemoryRef[]; // 由情景记忆溯源到的语义记忆
}

// 对话原文条目（来自 locomo 数据集的 conversation.session_*）
export interface ConversationMessage {
  speaker: string;
  dia_id: string; // 如 D1:3
  text: string;
  img_url?: string[];
  blip_caption?: string;
  query?: string;
}

// 前端构建的对话索引条目
export interface ConversationIndexEntry {
  user_key: string; // e.g. Caroline_0
  speakers: { a: string; b: string };
  diaIdToMessage: Record<string, ConversationMessage>;
}

// 轻量情景记忆引用
export interface EpisodeRef {
  episode_id: string;
  title?: string;
}

// 轻量语义记忆引用
export interface SemanticMemoryRef {
  memory_id: string;
  content: string;
  confidence?: number;
}
