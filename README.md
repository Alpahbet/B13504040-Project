# Discord Bot README

## (1) 程式的功能 Features

這個 Discord Bot 提供一個家、學校或任何組織維持溝通並管理日常計劃的工具：

1. **初始化每日計劃表**：使用者可以定義指定日期的計劃表。
2. **增加任務**：在指定時段增加任務。
3. **查看日計表**：計劃表以嵌入消息的形式顯示。
4. **查看月曆圖**：使用者可以查看月曆，並點註已經過去的日期。
5. **完成任務**：可將指定時段的任務標示為完成。

## (2) 使用方式 Usage

### 1. 啟動和使用指令

#### 啟動 Bot

在你的程式工具中執行 Python 程序，啟動 Bot，設定之前使用 Discord Developer Portal 創建 Bot 應用給予權限。

#### 指令介紹

- **`!start_plan YYYY-MM-DD`**: 初始化一個日期的計劃表，例如：`!start_plan 2024-12-20`
- **`!add_task YYYY-MM-DD TIME-PERIOD TASK`**: 在計劃表中增加任務，例如：`!add_task 2024-12-20 8-10時 調查資料`
- **`!view_plan YYYY-MM-DD`**: 顯示指定日期的計劃表，例如：`!view_plan 2024-12-20`
- **`!view_calendar YYYY-MM`**: 顯示指定月份的月曆視圖，例如：`!view_calendar 2024-12`
- **`!complete_task YYYY-MM-DD TIME-PERIOD`**: 將任務標示為完成，例如：`!complete_task 2024-12-20 8-10時`

## (3) 程式的架構 Program Architecture

<pre>
Discord-Bot/
├── main.py               # Bot 主程式入口
├── commands/
│   ├── plan.py           # 處理計畫相關指令
│   ├── calendar.py       # 處理月曆指令
├── data/
│   └── plans.json        # 儲存用戶計畫數據
├── utils/
│   └── helpers.py        # 工具函數和數據處理
├── requirements.txt      # 所需依賴
└── README.md             # 文件
</pre>


- **核心元件**：
  - `plan.py`：處理每日計劃相關邏輯，如初始化和新增任務。
  - `calendar.py`：提供月曆功能，展示視圖並支持交互。
  - `helpers.py`：包含數據讀寫和錯誤處理的輔助函數。
- **數據儲存**：使用 JSON 檔案進行本地化儲存，設有 `load_data()` 與 `save_data()` 功能便於讀寫。

## (4) 開發過程 Development Process

1. **基礎學習**：
   - 檢視 Discord.py 的基本文法和功能，學習設置 Bot 的 Intents 以及與伺服器進行互動。
2. **功能實現**：
   - 通過官方文檔和實測掌握 JSON 資料的存取與動態更新，設計了數據保存的架構。
   - 利用 `discord.Embed` 設計嵌入格式消息，用於日曆和每日計劃的顯示。
3. **問題解決**：
   - 解決多執行緒讀寫 JSON 導致數據覆蓋的問題，通過鎖機制和測試確保數據完整性。
4. **功能優化**：
   - 測試多種輸入錯誤情境，完善錯誤處理，包括日期格式錯誤和參數缺失的反饋。
5. **視覺設計**：
   - 在 ChatGPT 建議下，進一步優化日曆展示，包括過去日期的劃掉效果。

## (5) 參考資料來源 References

1. [Discord.py 文件網站](https://discordpy.readthedocs.io/)
2. ChatGPT 提供的邏輯和指令架構建議。
3. Python 官方文檔：學習 JSON 模組和 datetime 模組的使用。
4. 網絡資源：探索月曆生成的最佳實踐並進行改進。

## (6) 程式修正或增強的內容 Enhancements and Contributions

### 參考代碼

1. **樣式改進**：
   - 將 ChatGPT 提供的每日計畫時段格式調整為符合中文使用習慣的 "8-10時"。
2. **功能增強**：
   - 新增 `!complete_task` 功能，從邏輯到視覺效果（如✅ 標記）均由我設計實現。
3. **視覺增強**：
   - 基於 ChatGPT 的建議，使用 `discord.Embed` 添加色彩和格式調整。
   - 日曆中過去日期的任務添加劃掉樣式，以突出未來計劃。
4. **錯誤處理**：
   - 提供精確錯誤提示，例如日期格式錯誤或缺少參數。
   - 測試多種錯誤情境，使用使用順暢。
This README was made by ChatGPT.

