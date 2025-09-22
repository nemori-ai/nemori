import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import { TrendingUp, Target, Brain, Clock } from 'lucide-react';
import { EnhancedResultItem } from '../types';

interface StatsDashboardProps {
  data: EnhancedResultItem[];
}

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

export const StatsDashboard: React.FC<StatsDashboardProps> = ({ data }) => {
  // 计算整体统计
  const totalQuestions = data.length;
  const correctAnswers = data.filter(item => item.is_correct).length;
  const overallAccuracy = totalQuestions > 0 ? (correctAnswers / totalQuestions * 100) : 0;

  // 按类别统计
  const categoryStats = data.reduce((acc, item) => {
    const category = item.category || 'Unknown';
    if (!acc[category]) {
      acc[category] = { total: 0, correct: 0 };
    }
    acc[category].total++;
    if (item.is_correct) {
      acc[category].correct++;
    }
    return acc;
  }, {} as Record<string, { total: number; correct: number }>);

  const categoryChartData = Object.entries(categoryStats).map(([category, stats]) => ({
    category: `类别 ${category}`,
    accuracy: (stats.correct / stats.total * 100).toFixed(1),
    count: stats.total,
    correct: stats.correct
  }));

  // 分数分布统计
  const scoreRanges = {
    'F1 > 0.8': data.filter(item => (item.f1_score || 0) > 0.8).length,
    'F1 0.6-0.8': data.filter(item => {
      const f1 = item.f1_score || 0;
      return f1 > 0.6 && f1 <= 0.8;
    }).length,
    'F1 0.4-0.6': data.filter(item => {
      const f1 = item.f1_score || 0;
      return f1 > 0.4 && f1 <= 0.6;
    }).length,
    'F1 0.2-0.4': data.filter(item => {
      const f1 = item.f1_score || 0;
      return f1 > 0.2 && f1 <= 0.4;
    }).length,
    'F1 ≤ 0.2': data.filter(item => (item.f1_score || 0) <= 0.2).length
  };

  const scoreDistributionData = Object.entries(scoreRanges).map(([range, count]) => ({
    name: range,
    value: count
  }));

  // 平均分数
  const avgBleu = data.reduce((sum, item) => sum + (item.bleu_score || 0), 0) / totalQuestions;
  const avgF1 = data.reduce((sum, item) => sum + (item.f1_score || 0), 0) / totalQuestions;
  const avgMemoryTime = data.reduce((sum, item) => sum + (item.speaker_1_memory_time || 0), 0) / totalQuestions;

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">评测统计</h2>
      
      {/* Overall Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-gradient-to-r from-blue-500 to-blue-600 p-4 rounded-lg text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-blue-100 text-sm">整体准确率</p>
              <p className="text-2xl font-bold">{overallAccuracy.toFixed(1)}%</p>
              <p className="text-xs text-blue-100">{correctAnswers}/{totalQuestions}</p>
            </div>
            <Target className="w-8 h-8 text-blue-200" />
          </div>
        </div>
        
        <div className="bg-gradient-to-r from-green-500 to-green-600 p-4 rounded-lg text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-green-100 text-sm">平均 F1 分数</p>
              <p className="text-2xl font-bold">{avgF1.toFixed(3)}</p>
              <p className="text-xs text-green-100">语义重叠</p>
            </div>
            <TrendingUp className="w-8 h-8 text-green-200" />
          </div>
        </div>

        <div className="bg-gradient-to-r from-yellow-500 to-yellow-600 p-4 rounded-lg text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-yellow-100 text-sm">平均 BLEU 分数</p>
              <p className="text-2xl font-bold">{avgBleu.toFixed(3)}</p>
              <p className="text-xs text-yellow-100">文本相似度</p>
            </div>
            <Brain className="w-8 h-8 text-yellow-200" />
          </div>
        </div>

        <div className="bg-gradient-to-r from-purple-500 to-purple-600 p-4 rounded-lg text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-purple-100 text-sm">平均检索时间</p>
              <p className="text-2xl font-bold">{avgMemoryTime.toFixed(2)}s</p>
              <p className="text-xs text-purple-100">记忆查找</p>
            </div>
            <Clock className="w-8 h-8 text-purple-200" />
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Category Performance Chart */}
        <div>
          <h3 className="text-lg font-semibold text-gray-900 mb-4">各类别准确率</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={categoryChartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="category" />
              <YAxis domain={[0, 100]} />
              <Tooltip 
                formatter={(value, name) => [
                  name === 'accuracy' ? `${value}%` : value,
                  name === 'accuracy' ? '准确率' : '题目数'
                ]}
              />
              <Bar dataKey="accuracy" fill="#3b82f6" radius={4} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Score Distribution Pie Chart */}
        <div>
          <h3 className="text-lg font-semibold text-gray-900 mb-4">F1 分数分布</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={scoreDistributionData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({name, value}) => `${name}: ${value}`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {scoreDistributionData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Detailed Category Table */}
      <div className="mt-8">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">详细类别统计</h3>
        <div className="overflow-hidden shadow ring-1 ring-black ring-opacity-5 rounded-lg">
          <table className="min-w-full divide-y divide-gray-300">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  类别
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  准确率
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  正确/总数
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  平均F1
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  平均BLEU
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {Object.entries(categoryStats).map(([category, stats]) => {
                const categoryItems = data.filter(item => item.category === category);
                const avgCatF1 = categoryItems.reduce((sum, item) => sum + (item.f1_score || 0), 0) / categoryItems.length;
                const avgCatBleu = categoryItems.reduce((sum, item) => sum + (item.bleu_score || 0), 0) / categoryItems.length;
                const accuracy = (stats.correct / stats.total * 100);

                return (
                  <tr key={category} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      类别 {category}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      <div className="flex items-center">
                        <div className={`inline-flex px-2 py-1 rounded-full text-xs font-medium ${
                          accuracy >= 80 ? 'bg-green-100 text-green-800' :
                          accuracy >= 60 ? 'bg-yellow-100 text-yellow-800' :
                          'bg-red-100 text-red-800'
                        }`}>
                          {accuracy.toFixed(1)}%
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {stats.correct}/{stats.total}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {avgCatF1.toFixed(3)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {avgCatBleu.toFixed(3)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
