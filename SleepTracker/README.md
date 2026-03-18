# Sleep Tracker App - Advanced Analytics Edition

A professional Android app to monitor sleep patterns with real-time analytics, sleep quality charts, historical trends, and bedtime goal management. Features an intuitive analytics dashboard powered by MPAndroidChart.

## ✨ Features

✅ **Advanced Analytics Dashboard**
- Sleep quality line chart with trends
- Time period selection (Days, Weeks, Months, All)
- Summary statistics (best/worst night, total nights, avg quality)
- Last 7 nights comparison
- Trend indicator (improving/declining)

✅ **Real-time Sleep Tracking**
- Modal bottom sheet interface for tracking
- Start/Stop/Reset tracking controls
- Live elapsed time display in HH:MM:SS format
- Automatic sleep stage detection

✅ **Sleep Quality Scoring**
- 0-100% quality score based on duration
- Color-coded visual indicators
- Goal vs actual comparison
- Quality calculation algorithm

✅ **Data Persistence**
- Automatic saving of sleep records using SharedPreferences
- Historical data with daily summaries
- 7-day rolling averages
- Monthly trend aggregation

✅ **Sleep Stage Detection**
- **Awake**: Less than 30 minutes of sleep
- **Light Sleep**: Between 30-90 minutes
- **Deep Sleep**: 90+ minutes of sleep

✅ **Night-Friendly UI**
- Dark mode interface optimized for bedtime usage
- Low eye strain color palette
- Professional analytics visualization
- Smooth animations and transitions

## 🎯 App Overview

### Dashboard Screen (Main View)
- Professional line chart showing sleep quality trends
- Filter buttons to change time perspective
- Quick statistics summary
- Last 7 nights performance indicators
- Quick action buttons for key features

### Tracking Modal (Sleep Button)
- Clean tracking interface in bottom sheet
- Timer display for current session
- Sleep stage monitoring
- One-tap recording and saving

## 🛠️ Technology Stack

- **Language**: Kotlin
- **UI Framework**: Jetpack Compose + Material Design 3
- **Charts**: MPAndroidChart v3.1.0
- **Data Storage**: SharedPreferences (JSON serialization)
- **Min SDK**: API 28
- **Target SDK**: API 36
- **Compile SDK**: API 36

## 🏗️ Project Structure

```
SleepTracker/
├── app/
│   ├── src/main/
│   │   ├── java/com/example/sleeptracker/
│   │   │   ├── MainActivity.kt                 - App entry point
│   │   │   ├── data/
│   │   │   │   ├── SleepRecord.kt             - Sleep session model
│   │   │   │   ├── SleepTrackerService.kt     - Data persistence
│   │   │   │   └── AnalyticsService.kt        - Chart data aggregation
│   │   │   └── ui/
│   │   │       ├── theme/
│   │   │       │   ├── Color.kt               - Color palette
│   │   │       │   ├── Theme.kt               - Theme configuration
│   │   │       │   └── Type.kt                - Typography
│   │   │       └── screens/
│   │   │           ├── DashboardScreen.kt     - Analytics dashboard
│   │   │           ├── TrackingSheet.kt       - Sleep tracking modal
│   │   │           └── SettingsScreen.kt      - Settings (accessible)
│   │   └── utils/
│   │       └── ChartFormatter.kt              - Chart utilities
│   └── build.gradle.kts
└── gradle/
```

## 🚀 Building the App

### Prerequisites
- Android Studio (Arctic Fox or later)
- Android SDK API 36
- JDK 11 or later
- Gradle 8.0+

### Build Steps

1. **Open in Android Studio**
   ```
   File → Open → Select the SleepTracker directory
   ```

2. **Sync Gradle**
   - Android Studio will prompt to sync
   - Click "Sync Now"
   - MPAndroidChart will download automatically

3. **Build the App**
   ```
   Build → Make Project (Ctrl+F9 / Cmd+F9)
   ```

4. **Run on Device/Emulator**
   ```
   Run → Run 'app' (Shift+F10)
   ```

## 📊 Using the App

### Dashboard
1. **View Sleep Quality**: Line chart shows your sleep quality trends
2. **Change Time Period**: Click Days/Weeks/Months/All buttons
3. **Check Summary**: See best night, total nights, average quality
4. **Review Last 7 Nights**: Quick glance at past week performance

### Recording Sleep
1. **Click "Sleep" Button**: Opens tracking modal at bottom
2. **Press "Start"**: Begins tracking
3. **Monitor Stage**: Watch sleep stage change (Awake → Light → Deep)
4. **Press "Stop"**: Saves session with quality score
5. **Press "Reset"**: Clears without saving (for testing)

### Quick Actions
- **Meditation**: Placeholder for future meditation feature
- **Sleep**: Opens tracking modal
- **Journal**: Placeholder for note-taking
- **Stats**: Access detailed statistics
- **Profile**: User profile settings

## 📈 Quality Score System

```
Quality = (Sleep Duration / Goal Duration) × 100

Examples (with 8-hour goal):
- 4 hours → 50% quality (Red)
- 6 hours → 75% quality (Orange)
- 8 hours → 100% quality (Green)
- 10 hours → 100% quality (Green, capped)
```

### Color Coding
- 🟢 **Green**: 90%+ (Excellent)
- 🔵 **Blue**: 75-89% (Good)
- 🟠 **Orange**: 60-74% (Fair)
- 🔴 **Red**: <60% (Poor)

## 💾 Data Persistence

Sleep records stored with structure:
```json
{
  "startTime": 1234567890000,
  "endTime": 1234567890000,
  "duration": 28800000,
  "sleepStage": "Deep Sleep",
  "date": "2024-03-18",
  "qualityScore": 100,
  "notes": ""
}
```

- Duration and goals in milliseconds
- Dates in YYYY-MM-DD format
- Quality scores 0-100
- Data survives app restart

## 🧪 Testing the App

### With Sample Data
The app displays real data from your sleep records. To test:

1. **Record 5-10 sleep sessions** via the Sleep button
2. **Wait overnight** for new date entries
3. **Switch time periods** to see different views
4. **Verify quality scores** based on duration

### Without Data
Empty states show encouraging messages and sensible defaults.

## 🎨 Theme & Colors

Night-friendly color scheme designed for minimal eye strain:
- **Dark Background**: #1A1A1A
- **Dark Surface**: #2D2D2D
- **Light Text**: #FFFBFE
- **Accent Blue**: #64B5F6 (chart, highlights)
- **Deep Sleep**: #1E88E5 (premium blue)
- **Light Sleep**: #42A5F5 (light blue)
- **Awake**: #FF7043 (warm orange)
- **Goal Green**: #66BB6A (achievement color)

## 📚 Key Classes

### AnalyticsService
Calculates and aggregates sleep data:
- `getQualityScoresForPeriod()` - Period-based data
- `calculateQualityScore()` - Quality calculation
- `getLastSevenNightsData()` - Weekly comparison
- `getQuickStats()` - Summary statistics

### ChartFormatter
Formats data for charting:
- `convertToLineDataSet()` - Chart data conversion
- `getYAxisBounds()` - Dynamic scaling
- `getQualityColor()` - Color mapping

### SleepTrackerService
Manages data persistence:
- `saveSleepRecord()` - Save session
- `getAllRecords()` - Retrieve history
- `calculateAverageSleep()` - Average calculation

## ⚡ Performance

- **Chart Rendering**: Optimized for up to 365 data points
- **Data Aggregation**: Older data summarized into monthly values
- **Memory Efficient**: AndroidView with proper lifecycle
- **Fast Navigation**: Compose recomposition minimized

## 🐛 Troubleshooting

### App crashes on startup
- Clean build: `Build → Clean Project`
- Check Android SDK downloads
- Verify minSdk 28+ is installed

### Chart not displaying
- Ensure MPAndroidChart gradle dependency synced
- Check data exists (add sample records)
- Verify chart data points populate

### Data not saving
- Check SharedPreferences permissions
- Verify JSON serialization works
- Ensure Context passed correctly

### Tracking modal won't appear
- Verify Material3 ExperimentalApi annotations
- Check MainActivity ModalBottomSheet setup
- Confirm button click handlers work

## 📱 Device Requirements

- **Minimum**: Android 9 (API 28)
- **Target**: Android 15 (API 36)
- **RAM**: 50MB+ available
- **Storage**: 1MB for app + data

## 🔒 Privacy & Data

- All data stored locally on device
- No cloud synchronization
- No internet permission required
- Complete user control over data

## 📖 Implementation Details

For detailed implementation information, build instructions, and debugging tips:
- See `ADVANCED_UI_GUIDE.md` for complete guide
- Check inline code comments for logic explanation
- Refer to plan file: `.claude/plans/velvety-sparking-waffle.md`

## 🚀 Next Steps

1. **Build & Test** - Ensure compilation succeeds
2. **Sample Data** - Record several sessions for meaningful charts
3. **Verification** - Test all features per testing checklist
4. **Customization** - Adjust colors, text, or behavior as needed
5. **Enhancement** - Implement meditation, journal features

## 📝 License

This project is for educational purposes as part of the Application Development Lab course.

## 👨‍💻 Support

For issues or questions:
1. Check `ADVANCED_UI_GUIDE.md` troubleshooting section
2. Review inline code comments
3. Check the implementation plan
4. Review Android Studio error messages for guidance
