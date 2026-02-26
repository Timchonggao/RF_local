#!/usr/bin/env python3
import argparse
import datetime
import os
from collections import defaultdict

from flask import Flask, jsonify, render_template, request
from git import GitCommandError, Repo

app = Flask(__name__)

def analyze_commits(repo_path, days=7):
    """分析Git仓库最近N天的提交记录"""
    try:
        repo = Repo(repo_path)
        if repo.bare:
            raise ValueError("提供的路径是一个裸仓库，无法进行分析")

        # 计算截止日期（今天）和开始日期（N天前）
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)

        # 获取指定日期范围内的提交
        commits = []
        for commit in repo.iter_commits('HEAD'):
            commit_time = datetime.datetime.fromtimestamp(commit.authored_date)
            if commit_time >= start_date and commit_time <= end_date:
                commits.append(commit)
            # elif commit_time < start_date:
            #     break  # 已经超出指定日期范围，停止遍历

        return commits

    except GitCommandError as e:
        print(f"Git命令执行错误: {e}")
        return []
    except Exception as e:
        print(f"分析提交时出错: {e}")
        return []

def calculate_stats(repo_path, commits):
    """计算提交统计数据，包括代码量变化"""
    if not commits:
        return {
            'total_commits': 0,
            'commits_by_date': {},
            'commits_by_hour': defaultdict(int),
            'code_changes': {},
            'cumulative_additions': {},
            'cumulative_total': {}
        }

    # 总提交数
    total_commits = len(commits)

    # 按日期统计提交数
    commits_by_date = defaultdict(int)
    # 按小时统计提交数
    commits_by_hour = defaultdict(int)
    # 代码变更统计（添加/删除的行数）
    code_changes = defaultdict(lambda: {'additions': 0, 'deletions': 0, 'total': 0})

    # 获取仓库对象用于执行git diff命令
    repo = Repo(repo_path)

    for commit in commits:
        # 提交日期和时间
        commit_time = datetime.datetime.fromtimestamp(commit.authored_date)
        date_str = commit_time.strftime('%Y-%m-%d')
        hour = commit_time.hour

        # 按日期统计
        commits_by_date[date_str] += 1

        # 按小时统计
        commits_by_hour[hour] += 1

        # 计算代码变更（添加和删除的行数）
        try:
            # 获取当前提交的父提交
            if commit.parents:
                parent = commit.parents[0]
                # 使用git diff命令统计代码变更
                diff = repo.git.diff(parent.hexsha, commit.hexsha, '--numstat')

                additions = 0
                deletions = 0

                # 解析diff输出
                for line in diff.split('\n'):
                    if line.strip():
                        parts = line.split('\t')
                        if len(parts) >= 3:
                            try:
                                add = int(parts[0]) if parts[0] != '-' else 0
                                delete = int(parts[1]) if parts[1] != '-' else 0
                                additions += add
                                deletions += delete
                            except ValueError:
                                continue

                total_changes = additions + deletions
                code_changes[date_str]['additions'] += additions
                code_changes[date_str]['deletions'] += deletions
                code_changes[date_str]['total'] += total_changes
        except Exception as e:
            print(f"计算代码变更时出错: {e}")

    # 计算累积添加行数和累积总变更行数
    sorted_dates = sorted(code_changes.keys())
    cumulative_additions = {}
    cumulative_total = {}
    current_additions = 0
    current_total = 0

    for date in sorted_dates:
        current_additions += code_changes[date]['additions']
        current_total += code_changes[date]['total']
        cumulative_additions[date] = current_additions
        cumulative_total[date] = current_total

    # 转换为普通字典以便JSON序列化
    return {
        'total_commits': total_commits,
        'commits_by_date': dict(commits_by_date),
        'commits_by_hour': dict(commits_by_hour),
        'code_changes': dict(code_changes),
        'cumulative_additions': cumulative_additions,
        'cumulative_total': cumulative_total
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    """主页路由，处理表单提交和数据展示"""
    repo_path = app.config['REPO_PATH']
    default_days = app.config['DEFAULT_DAYS']

    if request.method == 'POST':
        days = int(request.form.get('days', default_days))
    else:
        days = default_days

    # 分析提交并计算统计数据
    commits = analyze_commits(repo_path, days)
    stats = calculate_stats(repo_path, commits)

    # 准备图表数据
    dates = sorted(stats['commits_by_date'].keys())
    commit_counts = [stats['commits_by_date'][date] for date in dates]

    hours = list(range(24))
    hourly_commit_counts = [stats['commits_by_hour'].get(hour, 0) for hour in hours]

    change_dates = sorted(stats['code_changes'].keys())
    additions = [stats['code_changes'][date]['additions'] for date in change_dates]
    deletions = [stats['code_changes'][date]['deletions'] for date in change_dates]
    total_changes = [stats['code_changes'][date]['total'] for date in change_dates]

    cumulative_additions = [stats['cumulative_additions'][date] for date in change_dates]
    cumulative_total = [stats['cumulative_total'][date] for date in change_dates]

    return render_template('dashboard.html',
                           repo_path=repo_path,
                           days=days,
                           total_commits=stats['total_commits'],
                           dates=dates,
                           commit_counts=commit_counts,
                           hours=hours,
                           hourly_commit_counts=hourly_commit_counts,
                           change_dates=change_dates,
                           additions=additions,
                           deletions=deletions,
                           total_changes=total_changes,
                           cumulative_additions=cumulative_additions,
                           cumulative_total=cumulative_total)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """API端点，返回JSON格式的统计数据"""
    repo_path = app.config['REPO_PATH']
    days = int(request.args.get('days', app.config['DEFAULT_DAYS']))

    commits = analyze_commits(repo_path, days)
    stats = calculate_stats(repo_path, commits)

    return jsonify(stats)

def create_templates_dir():
    """创建templates目录并添加HTML模板文件"""
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)

    # 模板文件内容
    template_content = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Git提交统计仪表盘</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdn.jsdelivr.net/npm/font-awesome@4.7.0/css/font-awesome.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.8/dist/chart.umd.min.js"></script>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        primary: '#3B82F6',
                        secondary: '#10B981',
                        accent: '#8B5CF6',
                        dark: '#1E293B',
                        light: '#F8FAFC'
                    },
                    fontFamily: {
                        sans: ['Inter', 'system-ui', 'sans-serif'],
                    },
                }
            }
        }
    </script>
    <style type="text/tailwindcss">
        @layer utilities {
            .content-auto {
                content-visibility: auto;
            }
            .card-shadow {
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            }
            .chart-container {
                transition: all 0.3s ease;
            }
            .chart-container:hover {
                transform: translateY(-5px);
            }
        }
    </style>
</head>
<body class="bg-gray-50 font-sans">
    <div class="min-h-screen flex flex-col">
        <!-- 顶部导航栏 -->
        <header class="bg-primary text-white shadow-md">
            <div class="container mx-auto px-4 py-3 flex justify-between items-center">
                <div class="flex items-center space-x-3">
                    <i class="fa fa-git-square text-2xl"></i>
                    <h1 class="text-xl font-bold">Git提交统计仪表盘</h1>
                </div>
                <div>
                    <span class="text-sm bg-white/20 px-3 py-1 rounded-full">
                        <i class="fa fa-folder-o mr-1"></i> {{ repo_path }}
                    </span>
                </div>
            </div>
        </header>

        <!-- 主要内容区 -->
        <main class="flex-grow container mx-auto px-4 py-6">
            <!-- 控制面板 -->
            <div class="bg-white rounded-lg shadow-lg p-6 mb-6 transform transition-all duration-300 hover:shadow-xl">
                <div class="flex flex-col md:flex-row justify-between items-center">
                    <div class="mb-4 md:mb-0">
                        <h2 class="text-xl font-bold text-gray-800">统计设置</h2>
                        <p class="text-gray-600">调整时间范围查看不同时间段的提交统计信息</p>
                    </div>
                    <form method="POST" class="flex items-center space-x-3">
                        <label for="days" class="text-gray-700 font-medium">最近</label>
                        <input type="number" id="days" name="days" min="1" max="365" 
                               value="{{ days }}" 
                               class="w-20 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary/50">
                        <span class="text-gray-700">天</span>
                        <button type="submit" class="bg-primary hover:bg-primary/90 text-white px-4 py-2 rounded-md transition-all duration-300 transform hover:scale-105 flex items-center">
                            <i class="fa fa-refresh mr-2"></i> 更新统计
                        </button>
                    </form>
                </div>
            </div>

            <!-- 统计概览 -->
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                <div class="bg-white rounded-lg shadow-md p-6 transform transition-all duration-300 hover:shadow-lg hover:bg-primary/5">
                    <div class="flex items-center">
                        <div class="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center text-primary">
                            <i class="fa fa-calendar-check-o text-xl"></i>
                        </div>
                        <div class="ml-4">
                            <h3 class="text-gray-600 font-medium">总提交数</h3>
                            <p class="text-3xl font-bold text-gray-800">{{ total_commits }}</p>
                        </div>
                    </div>
                </div>
                
                <div class="bg-white rounded-lg shadow-md p-6 transform transition-all duration-300 hover:shadow-lg hover:bg-secondary/5">
                    <div class="flex items-center">
                        <div class="w-12 h-12 rounded-full bg-secondary/10 flex items-center justify-center text-secondary">
                            <i class="fa fa-code-fork text-xl"></i>
                        </div>
                        <div class="ml-4">
                            <h3 class="text-gray-600 font-medium">平均每天提交</h3>
                            <p class="text-3xl font-bold text-gray-800">
                                {% if days > 0 %}{{ "%.2f"|format(total_commits/days) }}{% else %}0{% endif %}
                            </p>
                        </div>
                    </div>
                </div>
                
                <div class="bg-white rounded-lg shadow-md p-6 transform transition-all duration-300 hover:shadow-lg hover:bg-accent/5">
                    <div class="flex items-center">
                        <div class="w-12 h-12 rounded-full bg-accent/10 flex items-center justify-center text-accent">
                            <i class="fa fa-clock-o text-xl"></i>
                        </div>
                        <div class="ml-4">
                            <h3 class="text-gray-600 font-medium">统计周期</h3>
                            <p class="text-3xl font-bold text-gray-800">{{ days }} 天</p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- 图表区域 -->
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                <!-- 每日提交数量图表 -->
                <div class="bg-white rounded-lg shadow-md p-6 chart-container">
                    <h3 class="text-lg font-bold text-gray-800 mb-4 flex items-center">
                        <i class="fa fa-bar-chart text-primary mr-2"></i> 每日提交数量
                    </h3>
                    <div class="h-80">
                        <canvas id="commitsByDateChart"></canvas>
                    </div>
                </div>
                
                <!-- 提交时间分布图表 -->
                <div class="bg-white rounded-lg shadow-md p-6 chart-container">
                    <h3 class="text-lg font-bold text-gray-800 mb-4 flex items-center">
                        <i class="fa fa-clock-o text-secondary mr-2"></i> 提交时间分布
                    </h3>
                    <div class="h-80">
                        <canvas id="commitsByHourChart"></canvas>
                    </div>
                </div>
                
                <!-- 代码变更统计图表 -->
                <div class="bg-white rounded-lg shadow-md p-6 chart-container col-span-1 lg:col-span-2">
                    <h3 class="text-lg font-bold text-gray-800 mb-4 flex items-center">
                        <i class="fa fa-code text-accent mr-2"></i> 代码变更统计
                    </h3>
                    <div class="h-80">
                        <canvas id="codeChangesChart"></canvas>
                    </div>
                </div>
                
                <!-- 添加行数折线图 -->
                <div class="bg-white rounded-lg shadow-md p-6 chart-container col-span-1 lg:col-span-1">
                    <h3 class="text-lg font-bold text-gray-800 mb-4 flex items-center">
                        <i class="fa fa-line-chart text-secondary mr-2"></i> 添加行数趋势
                    </h3>
                    <div class="h-80">
                        <canvas id="additionsLineChart"></canvas>
                    </div>
                </div>
                
                <!-- 总变更行数折线图 -->
                <div class="bg-white rounded-lg shadow-md p-6 chart-container col-span-1 lg:col-span-1">
                    <h3 class="text-lg font-bold text-gray-800 mb-4 flex items-center">
                        <i class="fa fa-line-chart text-primary mr-2"></i> 总变更行数趋势
                    </h3>
                    <div class="h-80">
                        <canvas id="totalChangesLineChart"></canvas>
                    </div>
                </div>
                
                <!-- 累积添加行数图表 -->
                <div class="bg-white rounded-lg shadow-md p-6 chart-container col-span-1 lg:col-span-1">
                    <h3 class="text-lg font-bold text-gray-800 mb-4 flex items-center">
                        <i class="fa fa-area-chart text-secondary mr-2"></i> 累积添加行数
                    </h3>
                    <div class="h-80">
                        <canvas id="cumulativeAdditionsChart"></canvas>
                    </div>
                </div>
                
                <!-- 累积总变更行数图表 -->
                <div class="bg-white rounded-lg shadow-md p-6 chart-container col-span-1 lg:col-span-1">
                    <h3 class="text-lg font-bold text-gray-800 mb-4 flex items-center">
                        <i class="fa fa-area-chart text-primary mr-2"></i> 累积总变更行数
                    </h3>
                    <div class="h-80">
                        <canvas id="cumulativeTotalChart"></canvas>
                    </div>
                </div>
            </div>
        </main>

        <!-- 页脚 -->
        <footer class="bg-dark text-white py-4">
            <div class="container mx-auto px-4 text-center">
                <p>Git提交统计仪表盘 &copy; 2023</p>
            </div>
        </footer>
    </div>

    <script>
        // 初始化图表
        document.addEventListener('DOMContentLoaded', function() {
            // 每日提交数量图表
            const dateCtx = document.getElementById('commitsByDateChart').getContext('2d');
            const dateChart = new Chart(dateCtx, {
                type: 'bar',
                data: {
                    labels: {{ dates|tojson }},
                    datasets: [{
                        label: '提交数量',
                        data: {{ commit_counts|tojson }},
                        backgroundColor: 'rgba(59, 130, 246, 0.7)',
                        borderColor: 'rgba(59, 130, 246, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: {
                                color: 'rgba(0, 0, 0, 0.05)'
                            }
                        },
                        x: {
                            grid: {
                                display: false
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            backgroundColor: 'rgba(0, 0, 0, 0.7)',
                            padding: 10,
                            titleFont: {
                                size: 14,
                                weight: 'bold'
                            },
                            bodyFont: {
                                size: 13
                            }
                        }
                    },
                    animation: {
                        duration: 1500,
                        easing: 'easeOutQuart'
                    }
                }
            });

            // 提交时间分布图表
            const hourCtx = document.getElementById('commitsByHourChart').getContext('2d');
            const hourChart = new Chart(hourCtx, {
                type: 'line',
                data: {
                    labels: {{ hours|tojson }},
                    datasets: [{
                        label: '提交数量',
                        data: {{ hourly_commit_counts|tojson }},
                        backgroundColor: 'rgba(16, 185, 129, 0.2)',
                        borderColor: 'rgba(16, 185, 129, 1)',
                        borderWidth: 2,
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: {
                                color: 'rgba(0, 0, 0, 0.05)'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: '小时 (24小时制)'
                            },
                            grid: {
                                display: false
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            backgroundColor: 'rgba(0, 0, 0, 0.7)',
                            padding: 10,
                            titleFont: {
                                size: 14,
                                weight: 'bold'
                            },
                            bodyFont: {
                                size: 13
                            }
                        }
                    },
                    animation: {
                        duration: 1500,
                        easing: 'easeOutQuart'
                    }
                }
            });

            // 代码变更统计图表
            const changesCtx = document.getElementById('codeChangesChart').getContext('2d');
            const changesChart = new Chart(changesCtx, {
                type: 'bar',
                data: {
                    labels: {{ change_dates|tojson }},
                    datasets: [
                        {
                            label: '添加行数',
                            data: {{ additions|tojson }},
                            backgroundColor: 'rgba(16, 185, 129, 0.7)', // 绿色
                            borderColor: 'rgba(16, 185, 129, 1)',
                            borderWidth: 1
                        },
                        {
                            label: '删除行数',
                            data: {{ deletions|tojson }},
                            backgroundColor: 'rgba(239, 68, 68, 0.7)',
                            borderColor: 'rgba(239, 68, 68, 1)',
                            borderWidth: 1
                        },
                        {
                            label: '总变更行数',
                            data: {{ total_changes|tojson }},
                            backgroundColor: 'rgba(59, 130, 246, 0.7)', // 蓝色
                            borderColor: 'rgba(59, 130, 246, 1)',
                            borderWidth: 1
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: {
                                color: 'rgba(0, 0, 0, 0.05)'
                            }
                        },
                        x: {
                            grid: {
                                display: false
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            position: 'top',
                            labels: {
                                boxWidth: 12,
                                padding: 20
                            }
                        },
                        tooltip: {
                            backgroundColor: 'rgba(0, 0, 0, 0.7)',
                            padding: 10,
                            titleFont: {
                                size: 14,
                                weight: 'bold'
                            },
                            bodyFont: {
                                size: 13
                            }
                        }
                    },
                    animation: {
                        duration: 1500,
                        easing: 'easeOutQuart'
                    }
                }
            });

            // 添加行数折线图
            const additionsCtx = document.getElementById('additionsLineChart').getContext('2d');
            const additionsChart = new Chart(additionsCtx, {
                type: 'line',
                data: {
                    labels: {{ change_dates|tojson }},
                    datasets: [{
                        label: '添加行数',
                        data: {{ additions|tojson }},
                        backgroundColor: 'rgba(16, 185, 129, 0.1)', // 浅绿色
                        borderColor: 'rgba(16, 185, 129, 1)', // 绿色
                        borderWidth: 2,
                        tension: 0.3,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: {
                                color: 'rgba(0, 0, 0, 0.05)'
                            }
                        },
                        x: {
                            grid: {
                                display: false
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            backgroundColor: 'rgba(0, 0, 0, 0.7)',
                            padding: 10,
                            titleFont: {
                                size: 14,
                                weight: 'bold'
                            },
                            bodyFont: {
                                size: 13
                            }
                        }
                    },
                    animation: {
                        duration: 1500,
                        easing: 'easeOutQuart'
                    }
                }
            });

            // 总变更行数折线图
            const totalChangesCtx = document.getElementById('totalChangesLineChart').getContext('2d');
            const totalChangesChart = new Chart(totalChangesCtx, {
                type: 'line',
                data: {
                    labels: {{ change_dates|tojson }},
                    datasets: [{
                        label: '总变更行数',
                        data: {{ total_changes|tojson }},
                        backgroundColor: 'rgba(59, 130, 246, 0.1)', // 浅蓝色
                        borderColor: 'rgba(59, 130, 246, 1)', // 蓝色
                        borderWidth: 2,
                        tension: 0.3,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: {
                                color: 'rgba(0, 0, 0, 0.05)'
                            }
                        },
                        x: {
                            grid: {
                                display: false
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            backgroundColor: 'rgba(0, 0, 0, 0.7)',
                            padding: 10,
                            titleFont: {
                                size: 14,
                                weight: 'bold'
                            },
                            bodyFont: {
                                size: 13
                            }
                        }
                    },
                    animation: {
                        duration: 1500,
                        easing: 'easeOutQuart'
                    }
                }
            });

            // 累积添加行数图表
            const cumulativeAdditionsCtx = document.getElementById('cumulativeAdditionsChart').getContext('2d');
            const cumulativeAdditionsChart = new Chart(cumulativeAdditionsCtx, {
                type: 'line',
                data: {
                    labels: {{ change_dates|tojson }},
                    datasets: [{
                        label: '累积添加行数',
                        data: {{ cumulative_additions|tojson }},
                        backgroundColor: 'rgba(16, 185, 129, 0.2)', // 浅绿色
                        borderColor: 'rgba(16, 185, 129, 1)', // 绿色
                        borderWidth: 2,
                        tension: 0.2,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: {
                                color: 'rgba(0, 0, 0, 0.05)'
                            }
                        },
                        x: {
                            grid: {
                                display: false
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            backgroundColor: 'rgba(0, 0, 0, 0.7)',
                            padding: 10,
                            titleFont: {
                                size: 14,
                                weight: 'bold'
                            },
                            bodyFont: {
                                size: 13
                            }
                        }
                    },
                    animation: {
                        duration: 1500,
                        easing: 'easeOutQuart'
                    }
                }
            });

            // 累积总变更行数图表
            const cumulativeTotalCtx = document.getElementById('cumulativeTotalChart').getContext('2d');
            const cumulativeTotalChart = new Chart(cumulativeTotalCtx, {
                type: 'line',
                data: {
                    labels: {{ change_dates|tojson }},
                    datasets: [{
                        label: '累积总变更行数',
                        data: {{ cumulative_total|tojson }},
                        backgroundColor: 'rgba(59, 130, 246, 0.2)', // 浅蓝色
                        borderColor: 'rgba(59, 130, 246, 1)', // 蓝色
                        borderWidth: 2,
                        tension: 0.2,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: {
                                color: 'rgba(0, 0, 0, 0.05)'
                            }
                        },
                        x: {
                            grid: {
                                display: false
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            backgroundColor: 'rgba(0, 0, 0, 0.7)',
                            padding: 10,
                            titleFont: {
                                size: 14,
                                weight: 'bold'
                            },
                            bodyFont: {
                                size: 13
                            }
                        }
                    },
                    animation: {
                        duration: 1500,
                        easing: 'easeOutQuart'
                    }
                }
            });

            // 添加图表悬停效果
            const chartContainers = document.querySelectorAll('.chart-container');
            chartContainers.forEach(container => {
                container.addEventListener('mouseenter', () => {
                    container.style.boxShadow = '0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)';
                    container.style.transform = 'translateY(-5px)';
                });
                container.addEventListener('mouseleave', () => {
                    container.style.boxShadow = '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)';
                    container.style.transform = 'translateY(0)';
                });
            });
        });
    </script>
</body>
</html>
    '''

    # 写入模板文件
    with open(os.path.join(templates_dir, 'dashboard.html'), 'w', encoding='utf-8') as f:
        f.write(template_content)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Git提交统计与可视化工具')
    parser.add_argument('-r', '--repo', default='.', help='Git仓库路径（默认为当前目录）')
    parser.add_argument('-p', '--port', type=int, default=5000, help='Web服务器端口（默认为5000）')
    parser.add_argument('-d', '--days', type=int, default=7, help='默认统计最近多少天的提交（默认为7天）')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')

    args = parser.parse_args()

    # 检查仓库路径是否存在
    if not os.path.exists(args.repo):
        print(f"错误: 指定的仓库路径不存在: {args.repo}")
        return

    # 检查是否是Git仓库
    try:
        Repo(args.repo).git_dir
    except Exception:
        print(f"错误: 指定的路径不是一个有效的Git仓库: {args.repo}")
        return

    # 创建模板目录和文件
    create_templates_dir()

    # 配置Flask应用
    app.config['REPO_PATH'] = os.path.abspath(args.repo)
    app.config['DEFAULT_DAYS'] = args.days

    print("启动Git提交统计仪表盘...")
    print(f"仓库路径: {app.config['REPO_PATH']}")
    print(f"默认统计最近 {app.config['DEFAULT_DAYS']} 天的提交")
    print(f"访问 http://localhost:{args.port} 在浏览器中查看统计信息")

    # 启动Flask应用
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
