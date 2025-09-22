import React, { useState } from 'react';
import { Eye, EyeOff, Calendar, Hash, Star } from 'lucide-react';
import { Memory } from '../types';

interface MemoryViewerProps {
  memories: Memory[];
}

export const MemoryViewer: React.FC<MemoryViewerProps> = ({ memories }) => {
  const [showOriginalMessages, setShowOriginalMessages] = useState<Record<string, boolean>>({});

  const episodicMemories = memories.filter(m => m.memory_type === 'episodic');
  const semanticMemories = memories.filter(m => m.memory_type === 'semantic');

  const toggleOriginalMessages = (memoryId: string) => {
    setShowOriginalMessages(prev => ({
      ...prev,
      [memoryId]: !prev[memoryId]
    }));
  };

  const formatTimestamp = (timestamp: string) => {
    try {
      const date = new Date(timestamp);
      return date.toLocaleString('zh-CN');
    } catch {
      return timestamp;
    }
  };

  const renderMemoryItem = (memory: Memory, index: number) => (
    <div key={`${memory.episode_id}_${index}`} className={`memory-item ${memory.memory_type}`}>
      {/* Memory Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1">
          {/* Type Badge and Score */}
          <div className="flex items-center gap-2 mb-2">
            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
              memory.memory_type === 'episodic' 
                ? 'bg-blue-100 text-blue-800' 
                : 'bg-green-100 text-green-800'
            }`}>
              {memory.memory_type === 'episodic' ? '情景记忆' : '语义记忆'}
            </span>
            
            <div className="flex items-center gap-1 text-xs text-gray-500">
              <Star className="w-3 h-3" />
              <span>{memory.score.toFixed(3)}</span>
            </div>
            
            <span className="text-xs text-gray-400">
              {memory.search_method}
            </span>
          </div>

          {/* Episode Title (for episodic memories) */}
          {memory.episode_title && (
            <div className="flex items-center gap-1 text-sm text-gray-600 mb-2">
              <Hash className="w-3 h-3" />
              <span className="font-medium">{memory.episode_title}</span>
            </div>
          )}

          {/* Timestamp */}
          {memory.timestamp && (
            <div className="flex items-center gap-1 text-xs text-gray-500 mb-2">
              <Calendar className="w-3 h-3" />
              <span>{formatTimestamp(memory.timestamp)}</span>
            </div>
          )}
        </div>
      </div>

      {/* Memory Content */}
      <div className="prose prose-sm max-w-none">
        <p className="text-gray-700 leading-relaxed whitespace-pre-wrap">
          {memory.memory}
        </p>
      </div>

      {/* Additional Info for Semantic Memories */}
      {memory.memory_type === 'semantic' && (
        <div className="mt-3 pt-3 border-t border-gray-100">
          <div className="flex flex-wrap gap-4 text-xs text-gray-500">
            {memory.knowledge_type && (
              <span>知识类型: {memory.knowledge_type}</span>
            )}
            {memory.confidence && (
              <span>置信度: {memory.confidence.toFixed(2)}</span>
            )}
            {memory.related_episodes && memory.related_episodes.length > 0 && (
              <span>相关情景: {memory.related_episodes.length} 个</span>
            )}
          </div>
        </div>
      )}

      {/* Original Messages (for episodic memories) */}
      {memory.has_original_messages && memory.original_messages && memory.original_messages.length > 0 && (
        <div className="mt-4 pt-4 border-t border-gray-100">
          <button
            onClick={() => toggleOriginalMessages(memory.episode_id)}
            className="flex items-center gap-2 text-sm text-blue-600 hover:text-blue-700 font-medium"
          >
            {showOriginalMessages[memory.episode_id] ? (
              <EyeOff className="w-4 h-4" />
            ) : (
              <Eye className="w-4 h-4" />
            )}
            {showOriginalMessages[memory.episode_id] ? '隐藏' : '显示'} 原始对话 ({memory.original_messages.length} 条)
          </button>

          {showOriginalMessages[memory.episode_id] && (
            <div className="mt-3 bg-gray-50 rounded-lg p-3 max-h-60 overflow-y-auto custom-scrollbar">
              {memory.original_messages.map((msg, msgIndex) => (
                <div key={msgIndex} className="mb-2 last:mb-0">
                  <div className="flex items-start gap-2">
                    <span className="text-xs text-gray-400 flex-shrink-0 w-12">
                      {msg.role}:
                    </span>
                    <span className="text-xs text-gray-700 flex-1">
                      {msg.content}
                    </span>
                  </div>
                  {msg.timestamp && (
                    <div className="text-xs text-gray-400 ml-14 mt-1">
                      {formatTimestamp(msg.timestamp)}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );

  if (memories.length === 0) {
    return (
      <div className="bg-gray-50 rounded-lg p-6 text-center">
        <p className="text-gray-500">没有检索到相关记忆</p>
      </div>
    );
  }

  return (
    <div>
      <h4 className="font-medium text-gray-900 mb-4">
        检索到的记忆 ({memories.length} 条)
      </h4>
      
      {/* Summary */}
      <div className="flex gap-4 mb-4 text-sm text-gray-600">
        <span>情景记忆: {episodicMemories.length} 条</span>
        <span>语义记忆: {semanticMemories.length} 条</span>
      </div>

      {/* Memory List */}
      <div className="space-y-4">
        {/* Episodic Memories */}
        {episodicMemories.length > 0 && (
          <div>
            <h5 className="text-sm font-medium text-blue-700 mb-2">情景记忆</h5>
            {episodicMemories.map((memory, index) => renderMemoryItem(memory, index))}
          </div>
        )}

        {/* Semantic Memories */}
        {semanticMemories.length > 0 && (
          <div>
            <h5 className="text-sm font-medium text-green-700 mb-2">语义记忆</h5>
            {semanticMemories.map((memory, index) => renderMemoryItem(memory, index + episodicMemories.length))}
          </div>
        )}
      </div>
    </div>
  );
};
