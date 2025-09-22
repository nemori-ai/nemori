import React, { useMemo } from 'react';
import { AlertTriangle, CheckCircle2, Search, Link as LinkIcon } from 'lucide-react';
import { Memory, ConversationMessage, EpisodeRef, SemanticMemoryRef } from '../types';

interface EvidenceComparisonProps {
  question: string;
  retrievedMemories: Memory[];
  evidenceText: string[];
  evidencePointers?: string[];
  evidenceOriginalMessages?: ConversationMessage[];
  linkedEpisodes?: EpisodeRef[];
  linkedSemanticMemories?: SemanticMemoryRef[];
}

export const EvidenceComparison: React.FC<EvidenceComparisonProps> = ({
  question,
  retrievedMemories,
  evidenceText,
  evidencePointers,
  evidenceOriginalMessages,
  linkedEpisodes,
  linkedSemanticMemories
}) => {
  // 分析记忆与证据的匹配情况
  const analysis = useMemo(() => {
    const memoryTexts = retrievedMemories.map(m => m.memory.toLowerCase());
    
    const matchedEvidence: string[] = [];
    const missedEvidence: string[] = [];
    
    evidenceText.forEach(evidence => {
      const evidenceLower = evidence.toLowerCase();
      
      // 检查是否有任何记忆包含这个证据的关键词
      const isMatched = memoryTexts.some(memoryText => {
        // 简单的关键词匹配策略
        const evidenceWords = evidenceLower.split(/\s+/).filter(word => word.length > 3);
        const matchedWords = evidenceWords.filter(word => memoryText.includes(word));
        return matchedWords.length >= Math.min(2, evidenceWords.length * 0.5);
      });
      
      if (isMatched) {
        matchedEvidence.push(evidence);
      } else {
        missedEvidence.push(evidence);
      }
    });

    const coverageRate = evidenceText.length > 0 ? 
      (matchedEvidence.length / evidenceText.length * 100) : 0;

    return {
      matchedEvidence,
      missedEvidence,
      coverageRate,
      hasFullCoverage: missedEvidence.length === 0,
      totalEvidence: evidenceText.length
    };
  }, [retrievedMemories, evidenceText]);

  return (
    <div className="border border-gray-200 rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <h4 className="font-medium text-gray-900 flex items-center gap-2">
          <Search className="w-4 h-4" />
          证据覆盖分析
        </h4>
        
        <div className="flex items-center gap-2">
          {analysis.hasFullCoverage ? (
            <CheckCircle2 className="w-5 h-5 text-green-500" />
          ) : (
            <AlertTriangle className="w-5 h-5 text-yellow-500" />
          )}
          <span className={`text-sm font-medium ${
            analysis.coverageRate >= 80 ? 'text-green-600' :
            analysis.coverageRate >= 50 ? 'text-yellow-600' : 'text-red-600'
          }`}>
            {analysis.coverageRate.toFixed(1)}% 覆盖率
          </span>
        </div>
      </div>

      {/* Evidence Pointers and Originals */}
      {(evidencePointers && evidencePointers.length > 0) && (
        <div className="mb-4">
          <h5 className="text-sm font-medium text-gray-900 mb-2 flex items-center gap-2">
            <LinkIcon className="w-4 h-4" />
            证据定位标注
          </h5>
          <div className="flex flex-wrap gap-2">
            {evidencePointers.map((ptr, idx) => (
              <span key={idx} className="px-2 py-1 text-xs bg-gray-100 text-gray-800 rounded border border-gray-200">
                {ptr}
              </span>
            ))}
          </div>
          {evidenceOriginalMessages && evidenceOriginalMessages.length > 0 && (
            <div className="mt-3 space-y-2">
              {evidenceOriginalMessages.map((m, idx) => (
                <div key={idx} className="bg-gray-50 border border-gray-200 rounded p-3 text-sm">
                  <div className="text-gray-500 mb-1">{m.dia_id} · {m.speaker}</div>
                  <div className="text-gray-800">{m.text}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Linked Episodes and Semantic Memories */}
      {(linkedEpisodes && linkedEpisodes.length > 0) && (
        <div className="mb-4">
          <h5 className="text-sm font-medium text-gray-900 mb-2">关联情景记忆</h5>
          <div className="flex flex-wrap gap-2">
            {linkedEpisodes.map((ep) => (
              <span key={ep.episode_id} className="px-2 py-1 text-xs bg-blue-50 text-blue-800 rounded border border-blue-200">
                {ep.title || ep.episode_id}
              </span>
            ))}
          </div>
        </div>
      )}

      {(linkedSemanticMemories && linkedSemanticMemories.length > 0) && (
        <div className="mb-4">
          <h5 className="text-sm font-medium text-gray-900 mb-2">关联语义记忆</h5>
          <div className="space-y-2">
            {linkedSemanticMemories.map((sm) => (
              <div key={sm.memory_id} className="bg-purple-50 border border-purple-200 rounded p-3 text-sm">
                <div className="text-gray-800">{sm.content}</div>
                {sm.confidence !== undefined && (
                  <div className="text-xs text-gray-500 mt-1">confidence: {sm.confidence}</div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Coverage Summary */}
      <div className="grid grid-cols-3 gap-4 mb-4 text-sm">
        <div className="text-center p-2 bg-gray-50 rounded">
          <div className="font-medium text-gray-900">{analysis.totalEvidence}</div>
          <div className="text-gray-500">总证据数</div>
        </div>
        <div className="text-center p-2 bg-green-50 rounded">
          <div className="font-medium text-green-700">{analysis.matchedEvidence.length}</div>
          <div className="text-green-600">已检索到</div>
        </div>
        <div className="text-center p-2 bg-red-50 rounded">
          <div className="font-medium text-red-700">{analysis.missedEvidence.length}</div>
          <div className="text-red-600">遗漏</div>
        </div>
      </div>

      {/* Evidence Lists */}
      <div className="space-y-4">
        {/* Matched Evidence */}
        {analysis.matchedEvidence.length > 0 && (
          <div>
            <h5 className="text-sm font-medium text-green-700 mb-2 flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4" />
              成功检索的证据 ({analysis.matchedEvidence.length})
            </h5>
            <div className="space-y-2">
              {analysis.matchedEvidence.map((evidence, index) => (
                <div key={index} className="bg-green-50 border border-green-200 rounded p-3">
                  <p className="text-sm text-green-800">
                    {evidence}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Missed Evidence */}
        {analysis.missedEvidence.length > 0 && (
          <div>
            <h5 className="text-sm font-medium text-red-700 mb-2 flex items-center gap-2">
              <AlertTriangle className="w-4 h-4" />
              遗漏的证据 ({analysis.missedEvidence.length})
            </h5>
            <div className="space-y-2">
              {analysis.missedEvidence.map((evidence, index) => (
                <div key={index} className="bg-red-50 border border-red-200 rounded p-3">
                  <p className="text-sm text-red-800">{evidence}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Recommendations */}
      {analysis.missedEvidence.length > 0 && (
        <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded">
          <h6 className="text-sm font-medium text-yellow-800 mb-1">改进建议</h6>
          <ul className="text-xs text-yellow-700 space-y-1">
            <li>• 考虑扩大检索范围或调整检索算法</li>
            <li>• 检查遗漏证据是否存在于更早的对话中</li>
            <li>• 优化语义匹配算法以提高证据覆盖率</li>
          </ul>
        </div>
      )}
    </div>
  );
};
