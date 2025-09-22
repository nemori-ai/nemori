import React, { useState, useCallback } from 'react';
import { Upload, FileText, AlertCircle } from 'lucide-react';
import { ResultItem, MetricItem, DatasetItem, EnhancedResultItem } from '../types';

interface FileUploadProps {
  onDataProcessed: (data: EnhancedResultItem[]) => void;
  onError: (error: string) => void;
  onLoadingChange: (loading: boolean) => void;
}

export const FileUploadComponent: React.FC<FileUploadProps> = ({
  onDataProcessed,
  onError,
  onLoadingChange
}) => {
  const [resultsFile, setResultsFile] = useState<File | null>(null);
  const [metricsFile, setMetricsFile] = useState<File | null>(null);
  const [datasetFile, setDatasetFile] = useState<File | null>(null);

  const processFiles = useCallback(async () => {
    if (!resultsFile || !metricsFile) {
      onError('请选择results.json和metrics.json文件');
      return;
    }

    onLoadingChange(true);
    
    try {
      // 读取文件内容
      const [resultsText, metricsText, datasetText] = await Promise.all([
        resultsFile.text(),
        metricsFile.text(),
        datasetFile ? datasetFile.text() : Promise.resolve(null)
      ]);

      // 解析JSON
      const resultsData: Record<string, ResultItem[]> = JSON.parse(resultsText);
      const metricsData: Record<string, MetricItem[]> = JSON.parse(metricsText);
      const datasetData: DatasetItem[] = datasetText ? JSON.parse(datasetText) : [];

      // 扁平化results数据
      const flatResults: (ResultItem & { conversationId: string })[] = [];
      Object.entries(resultsData).forEach(([convId, items]) => {
        items.forEach(item => {
          flatResults.push({ ...item, conversationId: convId });
        });
      });

      // 扁平化metrics数据
      const flatMetrics: (MetricItem & { conversationId: string })[] = [];
      Object.entries(metricsData).forEach(([convId, items]) => {
        items.forEach(item => {
          flatMetrics.push({ ...item, conversationId: convId });
        });
      });

      // 创建问题到指标的映射
      const metricsMap = new Map<string, MetricItem>();
      flatMetrics.forEach(metric => {
        const key = `${metric.conversationId}_${metric.question}`;
        metricsMap.set(key, metric);
      });

      // 创建问题到证据的映射（如果有数据集）
      const evidenceMap = new Map<string, string[]>();
      if (datasetData.length > 0) {
        datasetData.forEach((item, idx) => {
          item.qa.forEach(qa => {
            const key = `${idx}_${qa.question}`;
            evidenceMap.set(key, qa.evidence || []);
          });
        });
      }

      // 合并数据
      const enhanced: EnhancedResultItem[] = flatResults.map(result => {
        const metricKey = `${result.conversationId}_${result.question}`;
        const metric = metricsMap.get(metricKey);
        
        const evidenceKey = `${result.conversationId}_${result.question}`;
        const evidence = evidenceMap.get(evidenceKey) || [];

        return {
          ...result,
          bleu_score: metric?.bleu_score,
          f1_score: metric?.f1_score, 
          llm_score: metric?.llm_score,
          evidence_text: evidence,
          conversation_id: result.conversationId,
          is_correct: metric?.llm_score === 1,
          has_evidence: evidence.length > 0,
          memory_count: (result.speaker_1_memories?.length || 0) + (result.speaker_2_memories?.length || 0)
        };
      });

      console.log(`处理完成: ${enhanced.length} 道题目`);
      onDataProcessed(enhanced);
      
    } catch (error) {
      console.error('文件处理错误:', error);
      onError(`文件处理失败: ${error instanceof Error ? error.message : '未知错误'}`);
    } finally {
      onLoadingChange(false);
    }
  }, [resultsFile, metricsFile, datasetFile, onDataProcessed, onError, onLoadingChange]);

  return (
    <div className="max-w-2xl mx-auto">
      <div className="bg-white rounded-lg shadow-lg p-8">
        <div className="text-center mb-8">
          <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
          <h2 className="text-2xl font-bold text-gray-900 mb-2">
            上传评测数据文件
          </h2>
          <p className="text-gray-600">
            请上传 results.json 和 metrics.json 文件开始分析
          </p>
        </div>

        <div className="space-y-6">
          {/* Results File Upload */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Results 文件 <span className="text-red-500">*</span>
            </label>
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 hover:border-primary-500 transition-colors">
              <input
                type="file"
                accept=".json"
                onChange={(e) => setResultsFile(e.target.files?.[0] || null)}
                className="w-full"
                id="results-upload"
              />
              {resultsFile && (
                <div className="mt-2 flex items-center text-sm text-green-600">
                  <FileText className="w-4 h-4 mr-2" />
                  {resultsFile.name}
                </div>
              )}
            </div>
          </div>

          {/* Metrics File Upload */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Metrics 文件 <span className="text-red-500">*</span>
            </label>
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 hover:border-primary-500 transition-colors">
              <input
                type="file"
                accept=".json"
                onChange={(e) => setMetricsFile(e.target.files?.[0] || null)}
                className="w-full"
                id="metrics-upload"
              />
              {metricsFile && (
                <div className="mt-2 flex items-center text-sm text-green-600">
                  <FileText className="w-4 h-4 mr-2" />
                  {metricsFile.name}
                </div>
              )}
            </div>
          </div>

          {/* Dataset File Upload (Optional) */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              数据集文件 (可选)
            </label>
            <div className="border-2 border-dashed border-gray-200 rounded-lg p-6 hover:border-primary-400 transition-colors">
              <input
                type="file"
                accept=".json"
                onChange={(e) => setDatasetFile(e.target.files?.[0] || null)}
                className="w-full"
                id="dataset-upload"
              />
              {datasetFile && (
                <div className="mt-2 flex items-center text-sm text-green-600">
                  <FileText className="w-4 h-4 mr-2" />
                  {datasetFile.name}
                </div>
              )}
              <p className="text-xs text-gray-500 mt-2">
                上传原始数据集以显示证据对比（包含evidence字段的文件）
              </p>
            </div>
          </div>

          {/* Process Button */}
          <button
            onClick={processFiles}
            disabled={!resultsFile || !metricsFile}
            className={`w-full py-3 px-4 rounded-lg font-medium transition-colors ${
              resultsFile && metricsFile
                ? 'bg-primary-600 text-white hover:bg-primary-700'
                : 'bg-gray-300 text-gray-500 cursor-not-allowed'
            }`}
          >
            开始分析数据
          </button>

          {/* Help Text */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex">
              <AlertCircle className="w-5 h-5 text-blue-600 mr-2 flex-shrink-0 mt-0.5" />
              <div className="text-sm text-blue-700">
                <h4 className="font-medium mb-1">使用说明：</h4>
                <ul className="space-y-1 text-xs">
                  <li>• Results文件：包含问题、答案、记忆等完整检索结果</li>
                  <li>• Metrics文件：包含各项评分指标（BLEU、F1、LLM Judge）</li>
                  <li>• 数据集文件：原始数据集，用于证据对比分析</li>
                  <li>• 支持的格式：标准LoCoMo评测JSON文件</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
