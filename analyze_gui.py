"""
불량별 이미지 분석 GUI 프로그램
tkinter를 사용한 그래픽 사용자 인터페이스
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
import threading
from typing import List, Optional, Dict
import matplotlib
matplotlib.use('TkAgg')  # 명시적으로 TkAgg 백엔드 사용
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.font_manager as fm
import numpy as np
import cv2
import platform

from defect_analyzer import DefectAnalyzer, Defect

# 한글 폰트 설정
def setup_korean_font():
    """한글 폰트 설정 - FontProperties 객체 반환"""
    system = platform.system()
    
    try:
        # 사용 가능한 폰트 목록 확인
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        # 한글 폰트 찾기 (우선순위 순)
        korean_fonts = []
        if system == 'Windows':
            korean_fonts = ['Malgun Gothic', '맑은 고딕', 'Gulim', 'Batang', 'Gungsuh', 'Dotum']
        elif system == 'Darwin':  # macOS
            korean_fonts = ['AppleGothic', 'Apple SD Gothic Neo']
        else:  # Linux
            korean_fonts = ['NanumGothic', 'NanumBarunGothic', 'NanumMyeongjo']
        
        found_font_name = None
        found_font_path = None
        
        # 폰트 이름으로 찾기
        for font_name in korean_fonts:
            if font_name in available_fonts:
                found_font_name = font_name
                # 폰트 경로 찾기
                for font in fm.fontManager.ttflist:
                    if font.name == font_name:
                        found_font_path = font.fname
                        break
                break
        
        # 폰트를 찾지 못한 경우 키워드로 검색
        if not found_font_name:
            korean_keywords = ['gothic', 'gulim', 'batang', 'nanum', 'malgun', 'apple']
            for font in fm.fontManager.ttflist:
                font_name_lower = font.name.lower()
                if any(keyword in font_name_lower for keyword in korean_keywords):
                    found_font_name = font.name
                    found_font_path = font.fname
                    break
        
        if found_font_name:
            # FontProperties 객체 생성
            if found_font_path:
                font_prop = fm.FontProperties(fname=found_font_path)
            else:
                font_prop = fm.FontProperties(family=found_font_name)
            
            # 전역 설정도 적용
            plt.rcParams['font.family'] = found_font_name
            plt.rcParams['axes.unicode_minus'] = False
            
            print(f"[DEBUG] 한글 폰트 설정 완료: {found_font_name}")
            if found_font_path:
                print(f"[DEBUG] 폰트 경로: {found_font_path}")
            
            return font_prop
        else:
            print("[DEBUG] 한글 폰트를 찾을 수 없습니다.")
            plt.rcParams['axes.unicode_minus'] = False
            return None
            
    except Exception as e:
        print(f"[DEBUG] 폰트 설정 오류: {e}")
        import traceback
        traceback.print_exc()
        plt.rcParams['axes.unicode_minus'] = False
        return None

# 프로그램 시작 시 한글 폰트 설정
KOREAN_FONT_PROP = setup_korean_font()

# 상수 정의
# 색상 맵 (matplotlib 색상)
DEFECT_COLOR_MAP = {
    'chipping': 'magenta',
    'crack': 'cyan',
    'scratch': 'orange',
    'point': 'red',
    'line': 'blue',
    'area': 'yellow',
    'edge': 'green'
}

# 색상 맵 (RGB 튜플)
DEFECT_COLOR_MAP_RGB = {
    'chipping': (255, 0, 255),    # magenta
    'crack': (0, 255, 255),       # cyan
    'scratch': (255, 165, 0),     # orange
    'point': (255, 0, 0),         # red
    'line': (0, 0, 255),          # blue
    'area': (255, 255, 0),        # yellow
    'edge': (0, 255, 0)           # green
}

# 이미지 초기화 텍스트
TEXT_ORIGINAL_IMAGE_PLACEHOLDER = '원본 이미지\n(이미지를 선택하세요)'
TEXT_RESULT_IMAGE_PLACEHOLDER = '분석 결과\n(분석을 실행하세요)'


class DefectAnalysisGUI:
    """불량 분석 GUI 애플리케이션"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("불량별 이미지 분석 시스템")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # 분석기 초기화
        self.analyzer = DefectAnalyzer()
        
        # 현재 상태
        self.current_image_path: Optional[Path] = None
        self.current_defects: Optional[List[Defect]] = None
        self.current_image: Optional[np.ndarray] = None
        
        # UI 설정
        self.show_labels = tk.BooleanVar(value=True)  # 라벨 표시 여부
        self.defect_label_visibility: Dict[str, tk.BooleanVar] = {}  # 각 불량명별 표시 여부
        self.defect_label_frame: Optional[ttk.Frame] = None  # 불량명 체크박스 프레임
        
        # GUI 구성
        self.create_widgets()
        
        # 스타일 설정
        self.setup_styles()
    
    def setup_styles(self):
        """스타일 설정"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # 버튼 스타일
        style.configure('Action.TButton', padding=10, font=('맑은 고딕', 10, 'bold'))
        style.configure('Primary.TButton', padding=8, font=('맑은 고딕', 9))
    
    def _add_text_with_font(self, ax, x, y, text, **kwargs):
        """
        한글 폰트를 지원하는 텍스트 추가 헬퍼 메서드
        
        Args:
            ax: matplotlib axes 객체
            x, y: 텍스트 위치
            text: 표시할 텍스트
            **kwargs: matplotlib text() 메서드의 추가 인자
        """
        if KOREAN_FONT_PROP:
            kwargs['fontproperties'] = KOREAN_FONT_PROP
        ax.text(x, y, text, **kwargs)
    
    def _set_title_with_font(self, ax, title, **kwargs):
        """
        한글 폰트를 지원하는 제목 설정 헬퍼 메서드
        
        Args:
            ax: matplotlib axes 객체
            title: 제목 텍스트
            **kwargs: matplotlib set_title() 메서드의 추가 인자
        """
        if KOREAN_FONT_PROP:
            kwargs['fontproperties'] = KOREAN_FONT_PROP
        ax.set_title(title, **kwargs)
    
    def _set_label_with_font(self, ax, label_type, text, **kwargs):
        """
        한글 폰트를 지원하는 라벨 설정 헬퍼 메서드
        
        Args:
            ax: matplotlib axes 객체
            label_type: 'xlabel', 'ylabel', 'title' 중 하나
            text: 라벨 텍스트
            **kwargs: matplotlib 라벨 메서드의 추가 인자
        """
        if KOREAN_FONT_PROP:
            kwargs['fontproperties'] = KOREAN_FONT_PROP
        
        if label_type == 'xlabel':
            ax.set_xlabel(text, **kwargs)
        elif label_type == 'ylabel':
            ax.set_ylabel(text, **kwargs)
        elif label_type == 'title':
            ax.set_title(text, **kwargs)
    
    def create_widgets(self):
        """위젯 생성"""
        # 메인 프레임
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # 왼쪽 패널 (설정)
        left_panel = ttk.LabelFrame(main_frame, text="분석 설정", padding="10")
        left_panel.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        left_panel.columnconfigure(0, weight=1)
        
        # 오른쪽 패널 (결과)
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(1, weight=1)
        
        # 이미지 선택 영역
        self.create_image_selection(left_panel)
        
        # 불량 유형 선택
        self.create_defect_type_selection(left_panel)
        
        # 분석 설정
        self.create_analysis_settings(left_panel)
        
        # 불량명 표시 설정
        self.create_defect_label_settings(left_panel)
        
        # 실행 버튼 (분석 설정 다음에 배치)
        self.create_action_buttons(left_panel)
        
        # 결과 표시 영역
        self.create_result_display(right_panel)
        
        # 상태바
        self.create_status_bar(main_frame)
    
    def create_image_selection(self, parent):
        """이미지 선택 영역 생성"""
        frame = ttk.LabelFrame(parent, text="이미지 선택", padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        frame.columnconfigure(0, weight=1)
        
        # 파일 선택 버튼
        btn_select_file = ttk.Button(
            frame, 
            text="📁 이미지 파일 선택",
            command=self.select_image_file,
            style='Primary.TButton'
        )
        btn_select_file.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # 폴더 선택 버튼
        btn_select_folder = ttk.Button(
            frame,
            text="📂 폴더 선택 (배치 처리)",
            command=self.select_folder,
            style='Primary.TButton'
        )
        btn_select_folder.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # 선택된 경로 표시
        self.label_image_path = ttk.Label(
            frame, 
            text="선택된 파일 없음",
            foreground="#666666",
            wraplength=200,
            font=('맑은 고딕', 9)
        )
        self.label_image_path.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # 배치 처리 모드
        self.batch_mode = tk.BooleanVar(value=False)
        self.check_batch = ttk.Checkbutton(
            frame,
            text="배치 처리 모드",
            variable=self.batch_mode
        )
        self.check_batch.grid(row=3, column=0, sticky=tk.W, pady=(10, 0))
    
    def create_defect_type_selection(self, parent):
        """불량 유형 선택 영역 생성"""
        frame = ttk.LabelFrame(parent, text="불량 유형 선택", padding="10")
        frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.defect_types = {
            'chipping': tk.BooleanVar(value=True),
            'crack': tk.BooleanVar(value=True),
            'scratch': tk.BooleanVar(value=True)
        }
        
        # 개별 불량 유형 (3개만)
        check_chipping = ttk.Checkbutton(
            frame,
            text="Chipping (가장자리/모서리)",
            variable=self.defect_types['chipping']
        )
        check_chipping.grid(row=0, column=0, sticky=tk.W, pady=2)
        
        check_crack = ttk.Checkbutton(
            frame,
            text="Crack (외곽 균열)",
            variable=self.defect_types['crack']
        )
        check_crack.grid(row=1, column=0, sticky=tk.W, pady=2)
        
        check_scratch = ttk.Checkbutton(
            frame,
            text="Scratch (선형 긁힘)",
            variable=self.defect_types['scratch']
        )
        check_scratch.grid(row=2, column=0, sticky=tk.W, pady=2)
    
    def create_analysis_settings(self, parent):
        """분석 설정 영역 생성"""
        frame = ttk.LabelFrame(parent, text="분석 설정", padding="10")
        frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        frame.columnconfigure(1, weight=1)
        
        # 최소 결함 크기
        ttk.Label(frame, text="최소 결함 크기:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.var_min_area = tk.IntVar(value=10)
        spin_min_area = ttk.Spinbox(
            frame,
            from_=1,
            to=10000,
            textvariable=self.var_min_area,
            width=15
        )
        spin_min_area.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 0))
        
        # 최대 결함 크기
        ttk.Label(frame, text="최대 결함 크기:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.var_max_area = tk.IntVar(value=100000)
        spin_max_area = ttk.Spinbox(
            frame,
            from_=100,
            to=1000000,
            textvariable=self.var_max_area,
            width=15
        )
        spin_max_area.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 0))
        
        # 이진화 방법
        ttk.Label(frame, text="이진화 방법:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.var_threshold = tk.StringVar(value='adaptive')
        combo_threshold = ttk.Combobox(
            frame,
            textvariable=self.var_threshold,
            values=['adaptive', 'otsu', 'manual'],
            state='readonly',
            width=12
        )
        combo_threshold.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 0))
    
    def create_defect_label_settings(self, parent):
        """불량명 표시 설정 영역 생성"""
        frame = ttk.LabelFrame(parent, text="불량명 표시 설정", padding="10")
        frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        frame.columnconfigure(0, weight=1)
        
        # 초기 메시지
        self.label_settings_msg = ttk.Label(
            frame,
            text="분석 후 불량명을 선택할 수 있습니다",
            foreground="#666666",
            wraplength=200,
            font=('맑은 고딕', 9)
        )
        self.label_settings_msg.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # 스크롤 가능한 프레임 (나중에 체크박스 추가)
        self.defect_label_frame = ttk.Frame(frame)
        self.defect_label_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.defect_label_frame.columnconfigure(0, weight=1)
    
    def update_defect_label_checkboxes(self):
        """불량명 토글 스위치 업데이트 (분석 결과에 따라 동적 생성)"""
        # 기존 위젯 제거
        if self.defect_label_frame:
            for widget in self.defect_label_frame.winfo_children():
                widget.destroy()
        
        # 기존 딕셔너리 초기화
        self.defect_label_visibility.clear()
        
        if not self.current_defects or len(self.current_defects) == 0:
            if self.label_settings_msg:
                self.label_settings_msg.config(
                    text="분석 후 불량명을 선택할 수 있습니다",
                    foreground="#666666"
                )
            return
        
        # 불량명별로 그룹화
        defect_labels = {}
        for defect in self.current_defects:
            label_key = f"{defect.defect_type.upper()} {defect.severity.upper()}"
            if label_key not in defect_labels:
                defect_labels[label_key] = []
            defect_labels[label_key].append(defect)
        
        # 메시지 업데이트
            if self.label_settings_msg:
                self.label_settings_msg.config(
                    text=f"감지된 불량명 ({len(defect_labels)}종류)",
                    foreground="#000000",
                    font=('맑은 고딕', 9, 'bold')
                )
        
        # 각 불량명에 대한 토글 스위치 생성
        row = 0
        for label_key in sorted(defect_labels.keys()):
            var = tk.BooleanVar(value=True)  # 기본값은 표시
            self.defect_label_visibility[label_key] = var
            
            # 토글 스위치를 포함한 프레임 생성
            switch_frame = ttk.Frame(self.defect_label_frame)
            switch_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=3)
            switch_frame.columnconfigure(1, weight=1)
            
            # 라벨 텍스트
            label_widget = ttk.Label(
                switch_frame,
                text=label_key,
                font=('맑은 고딕', 9, 'bold'),
                foreground="#333333"
            )
            label_widget.grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
            
            # 토글 스위치 생성
            toggle_switch = self.create_toggle_switch(
                switch_frame,
                var,
                command=self.on_label_visibility_changed
            )
            toggle_switch.grid(row=0, column=1, sticky=tk.E)
            
            row += 1
    
    def create_toggle_switch(self, parent, variable: tk.BooleanVar, command=None):
        """토글 스위치 위젯 생성"""
        switch_frame = ttk.Frame(parent)
        
        # Canvas로 토글 스위치 그리기
        # ttk 스타일에서 배경색 가져오기
        style = ttk.Style()
        bg_color = style.lookup('TFrame', 'background', default='#f0f0f0')
        
        canvas = tk.Canvas(
            switch_frame,
            width=50,
            height=24,
            highlightthickness=0,
            bg=bg_color,
            relief='flat',
            bd=0
        )
        canvas.pack()
        
        # 토글 스위치 상태에 따라 그리기
        def draw_switch():
            canvas.delete("all")
            is_on = variable.get()
            
            # 배경 (둥근 사각형)
            bg_color = "#4CAF50" if is_on else "#CCCCCC"
            canvas.create_oval(2, 2, 22, 22, fill=bg_color, outline="", tags="bg")
            canvas.create_oval(28, 2, 48, 22, fill=bg_color, outline="", tags="bg")
            canvas.create_rectangle(12, 2, 38, 22, fill=bg_color, outline="", tags="bg")
            
            # 스위치 원 (ON/OFF 위치)
            if is_on:
                canvas.create_oval(28, 4, 46, 20, fill="white", outline="", tags="switch")
            else:
                canvas.create_oval(4, 4, 22, 20, fill="white", outline="", tags="switch")
        
        # 초기 그리기
        draw_switch()
        
        # 클릭 이벤트
        def toggle():
            variable.set(not variable.get())
            draw_switch()
            if command:
                command()
        
        canvas.bind("<Button-1>", lambda e: toggle())
        canvas.bind("<Enter>", lambda e: canvas.config(cursor="hand2"))
        canvas.bind("<Leave>", lambda e: canvas.config(cursor=""))
        
        # 변수 변경 감지
        def on_var_change(*args):
            draw_switch()
        
        variable.trace_add("write", on_var_change)
        
        return switch_frame
    
    def on_label_visibility_changed(self):
        """불량명 표시 설정 변경 시 이미지 다시 그리기"""
        if self.current_defects and len(self.current_defects) > 0:
            self.display_image()
    
    def create_action_buttons(self, parent):
        """실행 버튼 영역 생성"""
        frame = ttk.Frame(parent)
        frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        frame.columnconfigure(0, weight=1)
        
        # 분석 실행 버튼
        self.btn_analyze = ttk.Button(
            frame,
            text="🔍 분석 실행",
            command=self.start_analysis,
            style='Action.TButton',
            state='disabled'
        )
        self.btn_analyze.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # 결과 저장 버튼
        self.btn_save = ttk.Button(
            frame,
            text="💾 결과 저장",
            command=self.save_results,
            style='Primary.TButton',
            state='disabled'
        )
        self.btn_save.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # 초기화 버튼
        self.btn_reset = ttk.Button(
            frame,
            text="🔄 초기화",
            command=self.reset_application,
            style='Primary.TButton'
        )
        self.btn_reset.grid(row=2, column=0, sticky=(tk.W, tk.E))
    
    def create_result_display(self, parent):
        """결과 표시 영역 생성"""
        # 탭 노트북 생성
        notebook = ttk.Notebook(parent)
        notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # 이미지 탭
        self.create_image_tab(notebook)
        
        # 통계 탭
        self.create_statistics_tab(notebook)
        
        # 리포트 탭
        self.create_report_tab(notebook)
    
    def create_image_tab(self, notebook):
        """이미지 표시 탭 생성"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="이미지 결과")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        
        # Matplotlib Figure 생성 (2개의 subplot: 원본, 결과)
        self.fig = Figure(figsize=(12, 6), dpi=100)
        # Figure 레벨에서 폰트 설정
        if KOREAN_FONT_PROP:
            self.fig.patch.set_facecolor('white')
        
        self.ax_original = self.fig.add_subplot(121)  # 왼쪽: 원본 이미지
        self.ax_result = self.fig.add_subplot(122)   # 오른쪽: 분석 결과
        
        # 초기 상태 설정
        self.ax_original.axis('off')
        self._add_text_with_font(
            self.ax_original, 0.5, 0.5, TEXT_ORIGINAL_IMAGE_PLACEHOLDER,
            ha='center', va='center', fontsize=12, color='gray',
            transform=self.ax_original.transAxes
        )
        
        self.ax_result.axis('off')
        self._add_text_with_font(
            self.ax_result, 0.5, 0.5, TEXT_RESULT_IMAGE_PLACEHOLDER,
            ha='center', va='center', fontsize=12, color='gray',
            transform=self.ax_result.transAxes
        )
        
        # Canvas 생성
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 스크롤바
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.canvas.get_tk_widget().yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
    
    def create_statistics_tab(self, notebook):
        """통계 탭 생성"""
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text="통계")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)
        
        # 통계 텍스트 영역
        self.text_stats = scrolledtext.ScrolledText(
            frame,
            wrap=tk.WORD,
            width=50,
            height=20,
            font=('맑은 고딕', 10)
        )
        self.text_stats.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # 차트 프레임
        chart_frame = ttk.Frame(frame)
        chart_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        
        self.fig_stats = Figure(figsize=(8, 4), dpi=100)
        self.ax_stats = self.fig_stats.add_subplot(111)
        
        self.canvas_stats = FigureCanvasTkAgg(self.fig_stats, master=chart_frame)
        self.canvas_stats.draw()
        self.canvas_stats.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    def create_report_tab(self, notebook):
        """리포트 탭 생성"""
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text="리포트")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        
        # 리포트 텍스트 영역
        self.text_report = scrolledtext.ScrolledText(
            frame,
            wrap=tk.WORD,
            width=50,
            height=30,
            font=('맑은 고딕', 9)
        )
        self.text_report.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    def create_status_bar(self, parent):
        """상태바 생성"""
        self.status_bar = ttk.Label(
            parent,
            text="준비",
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding="5"
        )
        self.status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))
    
    def select_image_file(self):
        """이미지 파일 선택"""
        file_path = filedialog.askopenfilename(
            title="이미지 파일 선택",
            filetypes=[
                ("이미지 파일", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("모든 파일", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = Path(file_path)
            self.label_image_path.config(
                text=f"파일: {self.current_image_path.name}",
                foreground="black"
            )
            self.batch_mode.set(False)
            self.check_batch.config(state='normal')
            self.btn_analyze.config(state='normal')
            self.update_status(f"이미지 선택됨: {self.current_image_path.name}")
            
            # 선택한 이미지 미리보기 표시
            self.display_original_image()
    
    def select_folder(self):
        """폴더 선택"""
        folder_path = filedialog.askdirectory(title="폴더 선택")
        
        if folder_path:
            self.current_image_path = Path(folder_path)
            self.label_image_path.config(
                text=f"폴더: {self.current_image_path.name}",
                foreground="black"
            )
            self.batch_mode.set(True)
            self.check_batch.config(state='disabled')
            self.btn_analyze.config(state='normal')
            self.update_status(f"폴더 선택됨: {self.current_image_path.name}")
    
    def get_selected_defect_types(self):
        """선택된 불량 유형 반환 (Chipping, Crack, Scratch만)"""
        selected = []
        
        for key, var in self.defect_types.items():
            if var.get():
                selected.append(key)
        
        # 최소 하나는 선택되어야 함 (기본값: 모두 선택)
        return selected if selected else ['chipping', 'crack', 'scratch']
    
    def start_analysis(self):
        """분석 시작"""
        if not self.current_image_path:
            messagebox.showwarning("경고", "이미지 파일 또는 폴더를 선택하세요.")
            return
        
        print(f"[DEBUG] 분석 시작: {self.current_image_path}")
        
        # 분석기 설정 업데이트
        try:
            self.analyzer = DefectAnalyzer(
                min_defect_area=self.var_min_area.get(),
                max_defect_area=self.var_max_area.get(),
                threshold_method=self.var_threshold.get()
            )
            print(f"[DEBUG] 분석기 생성 완료: min_area={self.var_min_area.get()}, max_area={self.var_max_area.get()}, threshold={self.var_threshold.get()}")
        except Exception as e:
            print(f"[DEBUG] 분석기 생성 오류: {e}")
            messagebox.showerror("오류", f"분석기 생성 중 오류 발생: {str(e)}")
            return
        
        # UI 비활성화
        self.btn_analyze.config(state='disabled')
        self.btn_save.config(state='disabled')
        self.update_status("분석 중...")
        
        # 별도 스레드에서 분석 실행
        try:
            if self.batch_mode.get():
                print("[DEBUG] 배치 모드로 분석 시작")
                thread = threading.Thread(target=self.analyze_batch, daemon=True)
            else:
                print("[DEBUG] 단일 이미지 모드로 분석 시작")
                thread = threading.Thread(target=self.analyze_single, daemon=True)
            thread.start()
            print(f"[DEBUG] 스레드 시작됨: {thread.is_alive()}")
        except Exception as e:
            print(f"[DEBUG] 스레드 시작 오류: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("오류", f"분석 시작 중 오류 발생: {str(e)}")
            self.btn_analyze.config(state='normal')
    
    def analyze_single(self):
        """단일 이미지 분석"""
        try:
            print(f"[DEBUG] analyze_single 시작: {self.current_image_path}")
            
            # 이미지가 이미 로드되어 있지 않으면 로드
            if self.current_image is None:
                print(f"[DEBUG] 이미지 로드 시도: {self.current_image_path}")
                image = cv2.imread(str(self.current_image_path))
                if image is None:
                    error_msg = f"이미지를 로드할 수 없습니다: {self.current_image_path}"
                    print(f"[DEBUG] {error_msg}")
                    self.root.after(0, lambda: messagebox.showerror("오류", error_msg))
                    self.root.after(0, lambda: self.btn_analyze.config(state='normal'))
                    return
                self.current_image = image
                print(f"[DEBUG] 이미지 로드 완료: shape={image.shape}")
            else:
                print(f"[DEBUG] 이미지 이미 로드됨: shape={self.current_image.shape}")
            
            # 분석 실행
            print("[DEBUG] analyze_image 호출 시작")
            all_defects = self.analyzer.analyze_image(str(self.current_image_path))
            print(f"[DEBUG] 분석 완료: {len(all_defects)}개의 불량 감지")
            
            # 불량 유형 필터링 (Chipping, Crack, Scratch만)
            selected_types = self.get_selected_defect_types()
            print(f"[DEBUG] 선택된 불량 유형: {selected_types}")
            
            # 선택된 유형만 필터링 (chipping, crack, scratch만)
            filtered_defects = [d for d in all_defects if d.defect_type in selected_types]
            
            self.current_defects = filtered_defects
            print(f"[DEBUG] 필터링 후 불량 수: {len(filtered_defects)}개")
            
            # 결과 표시 (불량이 없어도 표시)
            print("[DEBUG] display_results 호출 예정")
            self.root.after(0, self.display_results)
            print("[DEBUG] display_results 호출 완료")
            
        except Exception as e:
            import traceback
            error_msg = f"분석 중 오류 발생:\n{str(e)}"
            error_trace = traceback.format_exc()
            # 디버깅을 위해 콘솔에도 출력
            print(f"[DEBUG] 분석 오류: {error_msg}")
            print(f"[DEBUG] 트레이스백:\n{error_trace}")
            
            self.root.after(0, lambda: messagebox.showerror("오류", error_msg))
            self.root.after(0, lambda: self.update_status("오류 발생"))
            self.root.after(0, lambda: self.btn_analyze.config(state='normal'))
    
    def analyze_batch(self):
        """배치 분석"""
        try:
            # 이미지 파일 찾기
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            image_files = []
            for ext in image_extensions:
                image_files.extend(self.current_image_path.glob(f'*{ext}'))
                image_files.extend(self.current_image_path.glob(f'*{ext.upper()}'))
            
            if not image_files:
                self.root.after(0, lambda: messagebox.showwarning(
                    "경고", "폴더에서 이미지 파일을 찾을 수 없습니다."
                ))
                return
            
            total_defects = 0
            processed = 0
            
            for img_file in image_files:
                try:
                    defects = self.analyzer.analyze_image(str(img_file))
                    selected_types = self.get_selected_defect_types()
                    
                    # 선택된 유형만 필터링 (chipping, crack, scratch만)
                    filtered_defects = [d for d in defects if d.defect_type in selected_types]
                    
                    total_defects += len(filtered_defects)
                    processed += 1
                    
                    self.root.after(0, lambda p=processed, t=len(image_files): 
                                  self.update_status(f"처리 중: {p}/{t}"))
                    
                except Exception as e:
                    print(f"오류 ({img_file}): {e}")
            
            self.root.after(0, lambda: messagebox.showinfo(
                "완료", 
                f"배치 처리 완료!\n처리된 이미지: {processed}/{len(image_files)}\n총 감지된 불량: {total_defects}개"
            ))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("오류", f"배치 분석 중 오류 발생: {str(e)}"))
        
        finally:
            self.root.after(0, lambda: self.update_status("배치 처리 완료"))
            self.root.after(0, lambda: self.btn_analyze.config(state='normal'))
    
    def display_results(self):
        """결과 표시"""
        print("[DEBUG] display_results 시작")
        try:
            if self.current_image is None:
                print("[DEBUG] 이미지가 없음")
                messagebox.showwarning("경고", "이미지가 로드되지 않았습니다.")
                self.btn_analyze.config(state='normal')
                return
            
            defects_count = len(self.current_defects) if self.current_defects is not None else 0
            print(f"[DEBUG] 이미지 shape: {self.current_image.shape}, 불량 수: {defects_count}")
            
            # 불량이 없어도 결과를 표시해야 함
            # 이미지 표시
            print("[DEBUG] display_image 호출")
            self.display_image()
            print("[DEBUG] display_image 완료")
            
            # 통계 표시
            print("[DEBUG] display_statistics 호출")
            self.display_statistics()
            print("[DEBUG] display_statistics 완료")
            
            # 리포트 표시
            print("[DEBUG] display_report 호출")
            self.display_report()
            print("[DEBUG] display_report 완료")
            
            # 불량명 체크박스 업데이트
            print("[DEBUG] update_defect_label_checkboxes 호출")
            self.update_defect_label_checkboxes()
            print("[DEBUG] update_defect_label_checkboxes 완료")
            
            # 버튼 활성화
            self.btn_save.config(state='normal')
            self.btn_analyze.config(state='normal')
            
            if self.current_defects is not None and len(self.current_defects) > 0:
                status_msg = f"분석 완료: {len(self.current_defects)}개의 불량 감지"
            else:
                status_msg = "분석 완료: 불량이 감지되지 않았습니다."
            
            self.update_status(status_msg)
            print(f"[DEBUG] display_results 완료: {status_msg}")
            
        except Exception as e:
            import traceback
            error_msg = f"결과 표시 중 오류 발생:\n{str(e)}"
            error_trace = traceback.format_exc()
            print(f"[DEBUG] 결과 표시 오류: {error_msg}")
            print(f"[DEBUG] 트레이스백:\n{error_trace}")
            messagebox.showerror("오류", error_msg)
            self.btn_analyze.config(state='normal')
    
    def display_original_image(self):
        """원본 이미지 표시"""
        if not self.current_image_path or not self.current_image_path.is_file():
            return
        
        try:
            # 이미지 로드
            image = cv2.imread(str(self.current_image_path))
            if image is None:
                return
            
            self.current_image = image
            
            # 원본 이미지 표시 (왼쪽)
            self.ax_original.clear()
            self.ax_original.axis('off')
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.ax_original.imshow(image_rgb)
            self._set_title_with_font(self.ax_original, '원본 이미지', fontsize=12, fontweight='bold')
            
            # 결과 이미지 영역 초기화 (오른쪽)
            self.ax_result.clear()
            self.ax_result.axis('off')
            self._add_text_with_font(
                self.ax_result, 0.5, 0.5, TEXT_RESULT_IMAGE_PLACEHOLDER,
                ha='center', va='center', fontsize=12, color='gray',
                transform=self.ax_result.transAxes
            )
            
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            print(f"이미지 표시 오류: {e}")
    
    def display_image(self):
        """분석 결과 이미지 표시 (전/후 비교)"""
        if self.current_image is None:
            return
        
        # 원본 이미지 렌더링
        image_rgb = self._render_original_image()
        
        # 분석 결과 이미지 표시 (오른쪽)
        self.ax_result.clear()
        self.ax_result.axis('off')
        
        # 불량이 있는 경우에만 표시
        if self.current_defects is not None and len(self.current_defects) > 0:
            # 불량 영역만 강조하는 이미지 생성
            result_image = self.create_highlighted_image(image_rgb, self.current_defects)
            self.ax_result.imshow(result_image)
            
            # 불량 그리기 (바운딩 박스, 중심점)
            self._draw_defects_on_image()
            
            # 라벨 표시
            self._draw_defect_labels()
            
            self._set_title_with_font(
                self.ax_result, 
                f'분석 결과 (감지된 불량: {len(self.current_defects)}개)', 
                fontsize=12, fontweight='bold'
            )
        else:
            # 불량이 없는 경우 - 원본 이미지 표시
            self.ax_result.imshow(image_rgb)
            self._set_title_with_font(
                self.ax_result, '분석 결과 (불량 없음)', 
                fontsize=12, fontweight='bold'
            )
            self._add_text_with_font(
                self.ax_result, 0.5, 0.5, '불량이 감지되지 않았습니다',
                ha='center', va='center', fontsize=14, color='green',
                transform=self.ax_result.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7)
            )
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def _render_original_image(self) -> np.ndarray:
        """원본 이미지 렌더링"""
        self.ax_original.clear()
        self.ax_original.axis('off')
        image_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        self.ax_original.imshow(image_rgb)
        self._set_title_with_font(self.ax_original, '원본 이미지', fontsize=12, fontweight='bold')
        return image_rgb
    
    def _draw_defects_on_image(self):
        """불량 그리기 (바운딩 박스, 중심점)"""
        for defect in self.current_defects:
            x, y, w, h = defect.bbox
            color = DEFECT_COLOR_MAP.get(defect.defect_type, 'red')
            
            # 바운딩 박스 (얇게, 불량 영역을 가리지 않도록)
            rect = plt.Rectangle((x, y), w, h, linewidth=2, 
                               edgecolor=color, facecolor='none', alpha=0.7)
            self.ax_result.add_patch(rect)
            
            # 중심점 (작게)
            self.ax_result.plot(defect.centroid[0], defect.centroid[1], 'o', 
                        color=color, markersize=4, markeredgewidth=1, markeredgecolor='white', alpha=0.8)
    
    def _draw_defect_labels(self):
        """불량 라벨 표시"""
        if not self.show_labels.get():
            return
        
        for defect in self.current_defects:
            # 불량명 키 생성 (예: "CHIPPING MINOR", "CRACK MODERATE")
            label_key = f"{defect.defect_type.upper()} {defect.severity.upper()}"
            
            # 해당 불량명의 체크박스 상태 확인
            should_show = True
            if label_key in self.defect_label_visibility:
                should_show = self.defect_label_visibility[label_key].get()
            
            if should_show:
                x, y, w, h = defect.bbox
                color = DEFECT_COLOR_MAP.get(defect.defect_type, 'red')
                
                # 라벨 (작고 간결하게, 바운딩 박스 밖에 표시)
                label = f"{defect.defect_type.upper()}\n{defect.severity.upper()}"
                # 라벨 위치: 바운딩 박스 오른쪽 상단에 작게 표시
                label_x = x + w + 5  # 바운딩 박스 오른쪽에 약간 떨어뜨림
                label_y = y  # 상단에 맞춤
                
                self._add_text_with_font(
                    self.ax_result, label_x, label_y, label,
                    color=color, fontsize=7, fontweight='bold',
                    ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, 
                             edgecolor=color, linewidth=1)
                )
    
    def create_highlighted_image(self, image_rgb: np.ndarray, defects: List[Defect]) -> np.ndarray:
        """불량 영역만 강조된 이미지 생성"""
        # 배경을 어둡게 처리 (30% 밝기)
        darkened = (image_rgb * 0.3).astype(np.uint8)
        
        # 불량 영역 마스크 생성
        height, width = image_rgb.shape[:2]
        defect_mask = np.zeros((height, width), dtype=np.uint8)
        
        # 각 불량 영역을 마스크에 추가
        for defect in defects:
            x, y, w, h = defect.bbox
            # 바운딩 박스 영역을 약간 확장하여 더 잘 보이게
            x_start = max(0, x - 5)
            y_start = max(0, y - 5)
            x_end = min(width, x + w + 5)
            y_end = min(height, y + h + 5)
            
            defect_mask[y_start:y_end, x_start:x_end] = 255
        
        # 마스크를 3채널로 확장
        defect_mask_3ch = np.stack([defect_mask] * 3, axis=-1) / 255.0
        
        # 불량 영역은 원본 밝기로, 나머지는 어둡게
        result = np.where(defect_mask_3ch > 0, image_rgb, darkened)
        
        # 불량 영역에 색상 오버레이 추가 (약간의 투명도)
        # 각 불량 영역에 색상 오버레이
        for defect in defects:
            x, y, w, h = defect.bbox
            x_start = max(0, x - 5)
            y_start = max(0, y - 5)
            x_end = min(width, x + w + 5)
            y_end = min(height, y + h + 5)
            
            color = DEFECT_COLOR_MAP_RGB.get(defect.defect_type, (255, 0, 0))
            
            # 해당 영역에 색상 오버레이 (20% 투명도)
            overlay = result[y_start:y_end, x_start:x_end].copy()
            overlay_color = np.array(color, dtype=np.uint8)
            overlay_blend = (overlay * 0.8 + overlay_color * 0.2).astype(np.uint8)
            result[y_start:y_end, x_start:x_end] = overlay_blend
        
        return result.astype(np.uint8)
    
    def display_statistics(self):
        """통계 표시"""
        self.text_stats.delete(1.0, tk.END)
        
        if self.current_defects is None or len(self.current_defects) == 0:
            self.text_stats.insert(tk.END, "감지된 불량이 없습니다.\n\n이미지는 정상적으로 분석되었지만 불량이 발견되지 않았습니다.")
            
            # 차트 초기화
            self.ax_stats.clear()
            self._add_text_with_font(
                self.ax_stats, 0.5, 0.5, '불량 없음',
                ha='center', va='center', fontsize=14, color='green',
                transform=self.ax_stats.transAxes
            )
            self.fig_stats.tight_layout()
            self.canvas_stats.draw()
            return
        
        # 기본 통계
        stats_text = f"""
{'='*50}
분석 결과 통계
{'='*50}

총 감지된 불량: {len(self.current_defects)}개

[불량 유형별 분포]
"""
        
        type_counts = {}
        for defect in self.current_defects:
            defect_type = defect.defect_type
            type_counts[defect_type] = type_counts.get(defect_type, 0) + 1
        
        for defect_type, count in sorted(type_counts.items()):
            stats_text += f"  {defect_type}: {count}개\n"
        
        stats_text += "\n[심각도별 분포]\n"
        severity_counts = {}
        for defect in self.current_defects:
            severity = defect.severity
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        for severity, count in sorted(severity_counts.items()):
            stats_text += f"  {severity}: {count}개\n"
        
        stats_text += "\n[면적 통계]\n"
        areas = [d.area for d in self.current_defects]
        stats_text += f"  최소 면적: {min(areas):.2f} 픽셀\n"
        stats_text += f"  최대 면적: {max(areas):.2f} 픽셀\n"
        stats_text += f"  평균 면적: {sum(areas)/len(areas):.2f} 픽셀\n"
        stats_text += f"  총 면적: {sum(areas):.2f} 픽셀\n"
        
        self.text_stats.insert(tk.END, stats_text)
        
        # 차트 생성
        self.ax_stats.clear()
        
        if type_counts:
            types = list(type_counts.keys())
            counts = list(type_counts.values())
            
            self.ax_stats.bar(types, counts, color=['magenta', 'cyan', 'orange', 'purple', 'red', 'blue', 'yellow', 'green'][:len(types)])
            self._set_label_with_font(self.ax_stats, 'xlabel', '불량 유형')
            self._set_label_with_font(self.ax_stats, 'ylabel', '개수')
            self._set_label_with_font(self.ax_stats, 'title', '불량 유형별 분포')
            self.ax_stats.tick_params(axis='x', rotation=45)
            
            self.fig_stats.tight_layout()
            self.canvas_stats.draw()
    
    def display_report(self):
        """리포트 표시"""
        self.text_report.delete(1.0, tk.END)
        
        if self.current_defects is None or len(self.current_defects) == 0:
            report_text = f"{'='*70}\n"
            report_text += f"불량 분석 리포트\n"
            report_text += f"{'='*70}\n\n"
            report_text += f"이미지: {self.current_image_path.name if self.current_image_path else 'N/A'}\n"
            report_text += f"분석 일시: {self.get_current_time()}\n"
            report_text += f"총 불량 수: 0개\n\n"
            report_text += f"{'-'*70}\n"
            report_text += "감지된 불량이 없습니다.\n"
            report_text += "이미지는 정상적으로 분석되었지만 불량이 발견되지 않았습니다.\n"
            self.text_report.insert(tk.END, report_text)
            return
        
        # 리포트 헤더
        report_text = f"{'='*70}\n"
        report_text += f"불량 분석 리포트\n"
        report_text += f"{'='*70}\n\n"
        report_text += f"이미지: {self.current_image_path.name if self.current_image_path else 'N/A'}\n"
        report_text += f"분석 일시: {self.get_current_time()}\n"
        report_text += f"총 불량 수: {len(self.current_defects)}개\n\n"
        report_text += f"{'-'*70}\n"
        report_text += f"{'ID':<5} {'Type':<12} {'Severity':<10} {'Area':<12} {'Centroid':<20}\n"
        report_text += f"{'-'*70}\n"
        
        # 불량 상세 정보
        for defect in self.current_defects:
            report_text += f"{defect.id:<5} {defect.defect_type:<12} {defect.severity:<10} "
            report_text += f"{defect.area:<12.2f} ({defect.centroid[0]:.1f}, {defect.centroid[1]:.1f})\n"
        
        self.text_report.insert(tk.END, report_text)
    
    def get_current_time(self):
        """현재 시간 반환"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def save_results(self):
        """결과 저장"""
        if not self.current_image_path or self.current_defects is None or len(self.current_defects) == 0:
            messagebox.showwarning("경고", "저장할 결과가 없습니다.")
            return
        
        # 저장 폴더 선택
        save_dir = filedialog.askdirectory(title="결과 저장 폴더 선택")
        if not save_dir:
            return
        
        save_path = Path(save_dir)
        
        try:
            # 결과 이미지 저장
            result_img_path = save_path / f"{self.current_image_path.stem}_result.jpg"
            self.analyzer.visualize_results(
                str(self.current_image_path),
                self.current_defects,
                save_path=str(result_img_path),
                show=False
            )
            
            # 리포트 저장
            report_path = save_path / f"{self.current_image_path.stem}_report.csv"
            self.analyzer.generate_report(self.current_defects, str(report_path))
            
            messagebox.showinfo("완료", f"결과가 저장되었습니다:\n{result_img_path}\n{report_path}")
            self.update_status(f"결과 저장됨: {save_path}")
            
        except Exception as e:
            messagebox.showerror("오류", f"결과 저장 중 오류 발생: {str(e)}")
    
    def reset_application(self):
        """애플리케이션 초기화"""
        # 확인 대화상자
        if messagebox.askyesno("초기화 확인", "모든 설정과 결과를 초기화하시겠습니까?"):
            # 상태 초기화
            self._reset_application_state()
            
            # UI 초기화
            self._reset_application_ui()
            
            # 상태바 업데이트
            self.update_status("초기화 완료. 새로운 분석을 시작하세요.")
    
    def _reset_application_state(self):
        """애플리케이션 상태 초기화"""
        # 이미지 및 불량 데이터 초기화
        self.current_image_path = None
        self.current_defects = None
        self.current_image = None
        
        # 불량명 체크박스 초기화
        self.defect_label_visibility.clear()
        if self.defect_label_frame:
            for widget in self.defect_label_frame.winfo_children():
                widget.destroy()
        if self.label_settings_msg:
            self.label_settings_msg.config(
                text="분석 후 불량명을 선택할 수 있습니다",
                foreground="#666666"
            )
        
        # 불량 유형 초기화 (모두 선택 상태로)
        for key in ['chipping', 'crack', 'scratch']:
            self.defect_types[key].set(True)
        
        # 분석 설정 초기화
        self.var_min_area.set(10)
        self.var_max_area.set(100000)
        self.var_threshold.set('adaptive')
        
        # 배치 모드 초기화
        self.batch_mode.set(False)
    
    def _reset_application_ui(self):
        """애플리케이션 UI 초기화"""
        # 파일 경로 표시 초기화
        self.label_image_path.config(
            text="선택된 파일 없음",
            foreground="#666666"
        )
        self.check_batch.config(state='normal')
        
        # 버튼 상태 초기화
        self.btn_analyze.config(state='disabled')
        self.btn_save.config(state='disabled')
        
        # 이미지 영역 초기화
        self.ax_original.clear()
        self.ax_original.axis('off')
        self._add_text_with_font(
            self.ax_original, 0.5, 0.5, TEXT_ORIGINAL_IMAGE_PLACEHOLDER,
            ha='center', va='center', fontsize=12, color='gray',
            transform=self.ax_original.transAxes
        )
        
        self.ax_result.clear()
        self.ax_result.axis('off')
        self._add_text_with_font(
            self.ax_result, 0.5, 0.5, TEXT_RESULT_IMAGE_PLACEHOLDER,
            ha='center', va='center', fontsize=12, color='gray',
            transform=self.ax_result.transAxes
        )
        
        # 통계 초기화
        self.text_stats.delete(1.0, tk.END)
        self.text_stats.insert(tk.END, "분석 결과가 여기에 표시됩니다.")
        
        # 리포트 초기화
        self.text_report.delete(1.0, tk.END)
        self.text_report.insert(tk.END, "분석 리포트가 여기에 표시됩니다.")
        
        # 차트 초기화
        self.ax_stats.clear()
        self.canvas_stats.draw()
        
        # 캔버스 업데이트
        self.fig.tight_layout()
        self.canvas.draw()
    
    def update_status(self, message: str):
        """상태바 업데이트"""
        self.status_bar.config(text=message)


def main():
    """메인 함수"""
    root = tk.Tk()
    app = DefectAnalysisGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()

