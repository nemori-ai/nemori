import React, { useState, useMemo, useEffect } from 'react';
import { Search, Filter, ChevronDown, ChevronUp, CheckCircle, XCircle, Clock, Brain } from 'lucide-react';
import { EnhancedResultItem } from '../types';
import { MemoryViewer } from './MemoryViewer';
import { EvidenceComparison } from './EvidenceComparison';

interface QuestionExplorerProps {
  data: EnhancedResultItem[];
}

export const QuestionExplorer: React.FC<QuestionExplorerProps> = ({ data }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('');
  const [correctnessFilter, setCorrectnessFilter] = useState('');
  const [sortBy, setSortBy] = useState<'accuracy' | 'f1' | 'bleu' | 'memory_time'>('accuracy');
  const [expandedItems, setExpandedItems] = useState<Set<number>>(new Set());
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;

  // 默认自动展开第一条，便于发现新功能
  useEffect(() => {
    if (data.length > 0) {
      setExpandedItems(new Set([0]));
    }
  }, [data]);

  // 获取所有类别
  const categories = useMemo(() => {
    const cats = new Set(data.map(item => item.category).filter(Boolean));
    return Array.from(cats).sort();
  }, [data]);

  // 过滤和排序数据
  const filteredData = useMemo(() => {
    let filtered = data.filter(item => {
      const matchesSearch = searchTerm === '' || 
        item.question.toLowerCase().includes(searchTerm.toLowerCase()) ||
        item.answer.toLowerCase().includes(searchTerm.toLowerCase()) ||
        item.response.toLowerCase().includes(searchTerm.toLowerCase());
      
      const matchesCategory = selectedCategory === '' || item.category === selectedCategory;
      
      const matchesCorrectness = correctnessFilter === '' ||
        (correctnessFilter === 'correct' && item.is_correct) ||
        (correctnessFilter === 'incorrect' && !item.is_correct);

      return matchesSearch && matchesCategory && matchesCorrectness;
    });

    // 排序
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'accuracy':
          return (b.is_correct ? 1 : 0) - (a.is_correct ? 1 : 0);
        case 'f1':
          return (b.f1_score || 0) - (a.f1_score || 0);
        case 'bleu':
          return (b.bleu_score || 0) - (a.bleu_score || 0);
        case 'memory_time':
          return (b.speaker_1_memory_time || 0) - (a.speaker_1_memory_time || 0);
        default:
          return 0;
      }
    });

    return filtered;
  }, [data, searchTerm, selectedCategory, correctnessFilter, sortBy]);

  // 分页数据
  const paginatedData = useMemo(() => {
    const startIndex = (currentPage - 1) * itemsPerPage;
    return filteredData.slice(startIndex, startIndex + itemsPerPage);
  }, [filteredData, currentPage]);

  const totalPages = Math.ceil(filteredData.length / itemsPerPage);

  const toggleExpanded = (index: number) => {
    const newExpanded = new Set(expandedItems);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedItems(newExpanded);
  };

  const getScoreBadgeColor = (score: number) => {
    if (score >= 0.8) return 'bg-green-100 text-green-800 border-green-200';
    if (score >= 0.6) return 'bg-yellow-100 text-yellow-800 border-yellow-200';
    return 'bg-red-100 text-red-800 border-red-200';
  };

  return (
    <div className="bg-white rounded-lg shadow-lg">
      {/* Search and Filter Controls */}
      <div className="p-6 border-b border-gray-200">
        <div className="flex flex-col sm:flex-row gap-4">
          {/* Search */}
          <div className="flex-1">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
              <input
                type="text"
                placeholder="搜索问题、答案或回复..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10 pr-4 py-2 w-full border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              />
            </div>
          </div>
          
          {/* Category Filter */}
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
          >
            <option value="">所有类别</option>
            {categories.map(cat => (
              <option key={cat} value={cat}>类别 {cat}</option>
            ))}
          </select>

          {/* Correctness Filter */}
          <select
            value={correctnessFilter}
            onChange={(e) => setCorrectnessFilter(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
          >
            <option value="">正确性</option>
            <option value="correct">仅正确</option>
            <option value="incorrect">仅错误</option>
          </select>

          {/* Sort */}
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500"
          >
            <option value="accuracy">按准确性</option>
            <option value="f1">按F1分数</option>
            <option value="bleu">按BLEU分数</option>
            <option value="memory_time">按检索时间</option>
          </select>
        </div>

        {/* Results Count */}
        <div className="mt-4 flex items-center justify-between text-sm text-gray-500">
          <span>
            显示 {filteredData.length} 个结果，共 {data.length} 道题目
          </span>
          <span>
            准确率: {filteredData.length > 0 ? 
              (filteredData.filter(item => item.is_correct).length / filteredData.length * 100).toFixed(1) 
              : 0}%
          </span>
        </div>
      </div>

      {/* Results List */}
      <div className="divide-y divide-gray-200">
        {paginatedData.map((item, index) => {
          const globalIndex = (currentPage - 1) * itemsPerPage + index;
          const isExpanded = expandedItems.has(globalIndex);

          return (
            <div key={`${item.conversation_id}_${index}`} className="p-6">
              {/* Question Header */}
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  {/* Status and Category */}
                  <div className="flex items-center gap-2 mb-2">
                    {item.is_correct ? (
                      <CheckCircle className="w-5 h-5 text-green-500" />
                    ) : (
                      <XCircle className="w-5 h-5 text-red-500" />
                    )}
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                      类别 {item.category}
                    </span>
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                      对话 {item.conversation_id}
                    </span>
                  </div>

                  {/* Question */}
                  <h3 className="text-lg font-medium text-gray-900 mb-2">
                    {item.question}
                  </h3>

                  {/* Answer Comparison */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                    <div>
                      <p className="text-sm font-medium text-gray-700 mb-1">标准答案</p>
                      <p className="text-sm text-gray-600 bg-gray-50 p-2 rounded">
                        {item.answer}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm font-medium text-gray-700 mb-1">生成答案</p>
                      <p className={`text-sm p-2 rounded ${
                        item.is_correct ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'
                      }`}>
                        {item.response}
                      </p>
                    </div>
                  </div>

                  {/* Scores */}
                  <div className="flex flex-wrap gap-4">
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-gray-500">LLM Judge:</span>
                      {typeof item.is_correct === 'boolean' ? (
                        <span className={`px-2 py-1 rounded text-xs font-medium border ${
                          item.is_correct ? 'status-correct' : 'status-incorrect'
                        }`}>
                          {item.is_correct ? '正确' : '错误'}
                        </span>
                      ) : (
                        <span className="px-2 py-1 rounded text-xs font-medium border bg-gray-100 text-gray-700 border-gray-200">
                          未知
                        </span>
                      )}
                    </div>
                    {item.f1_score !== undefined && (
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-gray-500">F1:</span>
                        <span className={`px-2 py-1 rounded text-xs font-medium border ${
                          getScoreBadgeColor(item.f1_score)
                        }`}>
                          {item.f1_score.toFixed(3)}
                        </span>
                      </div>
                    )}
                    {item.bleu_score !== undefined && (
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-gray-500">BLEU:</span>
                        <span className={`px-2 py-1 rounded text-xs font-medium border ${
                          getScoreBadgeColor(item.bleu_score)
                        }`}>
                          {item.bleu_score.toFixed(3)}
                        </span>
                      </div>
                    )}
                    <div className="flex items-center gap-2">
                      <Clock className="w-3 h-3 text-gray-400" />
                      <span className="text-xs text-gray-500">
                        检索: {(item.speaker_1_memory_time || 0).toFixed(2)}s
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Brain className="w-3 h-3 text-gray-400" />
                      <span className="text-xs text-gray-500">
                        记忆: {item.memory_count || 0} 条
                      </span>
                    </div>
                  </div>
                </div>

                {/* Evidence Pointers (inline, always visible if present) */}
                {item.evidence_pointers && item.evidence_pointers.length > 0 && (
                  <div className="mt-3 flex items-center flex-wrap gap-2">
                    <span className="text-xs text-gray-500">证据:</span>
                    {item.evidence_pointers.map((ptr, idx) => (
                      <span
                        key={idx}
                        className="px-2 py-0.5 rounded text-xs bg-gray-100 text-gray-800 border border-gray-200"
                      >
                        {ptr}
                      </span>
                    ))}
                    <button
                      onClick={() => toggleExpanded(globalIndex)}
                      className="text-xs text-primary-600 hover:underline ml-2"
                    >
                      {isExpanded ? '收起原文' : '查看原文'}
                    </button>
                  </div>
                )}

                {/* Expand Button */}
                <button
                  onClick={() => toggleExpanded(globalIndex)}
                  className="ml-4 p-2 text-gray-400 hover:text-gray-600 transition-colors"
                >
                  {isExpanded ? (
                    <ChevronUp className="w-5 h-5" />
                  ) : (
                    <ChevronDown className="w-5 h-5" />
                  )}
                </button>
              </div>

              {/* Expanded Content */}
              {isExpanded && (
                <div className="mt-6 space-y-6 fade-in">
                  {/* Memory Viewer */}
                  <MemoryViewer 
                    memories={[...(item.speaker_1_memories || []), ...(item.speaker_2_memories || [])]}
                  />

                  {/* Evidence Comparison (show if any source available) */}
                  {(item.evidence_text?.length || 0) > 0 || (item.evidence_original_messages?.length || 0) > 0 ? (
                    <EvidenceComparison 
                      question={item.question}
                      retrievedMemories={[...(item.speaker_1_memories || []), ...(item.speaker_2_memories || [])]}
                      evidenceText={item.evidence_text || []}
                      evidencePointers={item.evidence_pointers}
                      evidenceOriginalMessages={item.evidence_original_messages}
                      linkedEpisodes={item.linked_episodes}
                      linkedSemanticMemories={item.linked_semantic_memories}
                    />
                  ) : null}

                  {/* Technical Details */}
                  <div className="bg-gray-50 rounded-lg p-4">
                    <h4 className="font-medium text-gray-900 mb-2">技术指标</h4>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <span className="text-gray-500">检索耗时:</span>
                        <span className="ml-2 font-medium">
                          {(item.speaker_1_memory_time || 0).toFixed(3)}s
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-500">回答耗时:</span>
                        <span className="ml-2 font-medium">
                          {(item.response_time || 0).toFixed(3)}s
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-500">记忆总数:</span>
                        <span className="ml-2 font-medium">
                          {item.memory_count}
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-500">系统版本:</span>
                        <span className="ml-2 font-medium text-xs">
                          {item.system_version || 'unknown'}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="px-6 py-4 border-t border-gray-200">
          <div className="flex items-center justify-between">
            <div className="text-sm text-gray-700">
              显示第 {(currentPage - 1) * itemsPerPage + 1} - {Math.min(currentPage * itemsPerPage, filteredData.length)} 项，
              共 {filteredData.length} 项结果
            </div>
            
            <div className="flex items-center gap-2">
              <button
                onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                disabled={currentPage === 1}
                className="px-3 py-2 text-sm font-medium text-gray-500 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                上一页
              </button>
              
              <span className="px-3 py-2 text-sm font-medium text-gray-700">
                第 {currentPage} 页，共 {totalPages} 页
              </span>
              
              <button
                onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                disabled={currentPage === totalPages}
                className="px-3 py-2 text-sm font-medium text-gray-500 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                下一页
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Empty State */}
      {filteredData.length === 0 && (
        <div className="p-12 text-center text-gray-500">
          <Filter className="w-12 h-12 mx-auto mb-4 text-gray-300" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">没有找到匹配的结果</h3>
          <p>请尝试调整搜索条件或筛选选项</p>
        </div>
      )}
    </div>
  );
};
