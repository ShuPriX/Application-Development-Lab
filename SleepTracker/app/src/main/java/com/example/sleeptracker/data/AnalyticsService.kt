package com.example.sleeptracker.data

import java.text.SimpleDateFormat
import java.util.*

data class QualityDataPoint(val date: String, val quality: Float)
data class DailyComparison(val date: String, val actualSleep: String, val goal: String, val metGoal: Boolean)
data class QuickStatsData(val bestNight: String, val worstNight: String, val totalNights: Int, val avgQuality: Float)

class AnalyticsService(private val sleepTrackerService: SleepTrackerService) {

    sealed class TimePeriod {
        object Days : TimePeriod()
        object Weeks : TimePeriod()
        object Months : TimePeriod()
        object All : TimePeriod()
    }

    /**
     * Get quality scores for a given time period
     */
    fun getQualityScoresForPeriod(period: TimePeriod): List<QualityDataPoint> {
        val allRecords = sleepTrackerService.getAllRecords()
        if (allRecords.isEmpty()) return emptyList()

        return when (period) {
            TimePeriod.Days -> getQualityForDays(allRecords, 30)
            TimePeriod.Weeks -> getQualityForWeeks(allRecords, 12)
            TimePeriod.Months -> getQualityForMonths(allRecords, 12)
            TimePeriod.All -> getQualityForAllTime(allRecords)
        }
    }

    /**
     * Calculate quality score for a single sleep record
     */
    fun calculateQualityScore(duration: Long, goalMs: Long): Int {
        val percentage = ((duration.toFloat() / goalMs.toFloat()) * 100).toInt()
        return minOf(100, maxOf(0, percentage))
    }

    /**
     * Get last 7 nights comparison data
     */
    fun getLastSevenNightsData(): List<DailyComparison> {
        val records = sleepTrackerService.getAllRecords()
        if (records.isEmpty()) return emptyList()

        val goal = sleepTrackerService.getBedtimeGoal()
        val dateFormat = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault())
        val comparisons = mutableListOf<DailyComparison>()

        // Get last 7 days
        val today = Calendar.getInstance()
        for (i in 6 downTo 0) {
            val date = Calendar.getInstance().apply {
                add(Calendar.DAY_OF_YEAR, -i)
            }
            val dateStr = dateFormat.format(date.time)
            val dayRecords = records.filter { it.date == dateStr }

            val totalDuration = dayRecords.sumOf { it.duration }
            val metGoal = totalDuration >= goal

            comparisons.add(
                DailyComparison(
                    date = dateStr,
                    actualSleep = formatDuration(totalDuration),
                    goal = formatDuration(goal),
                    metGoal = metGoal
                )
            )
        }

        return comparisons
    }

    /**
     * Get quick stats for dashboard
     */
    fun getQuickStats(): QuickStatsData {
        val records = sleepTrackerService.getAllRecords()
        if (records.isEmpty()) {
            return QuickStatsData(
                bestNight = "N/A",
                worstNight = "N/A",
                totalNights = 0,
                avgQuality = 0f
            )
        }

        val goal = sleepTrackerService.getBedtimeGoal()
        val qualities = records.map { calculateQualityScore(it.duration, goal).toFloat() }

        val bestDuration = records.maxByOrNull { it.duration }?.duration ?: 0L
        val worstDuration = records.minByOrNull { it.duration }?.duration ?: 0L

        return QuickStatsData(
            bestNight = formatDuration(bestDuration),
            worstNight = formatDuration(worstDuration),
            totalNights = records.size,
            avgQuality = qualities.average().toFloat()
        )
    }

    /**
     * Get trend data (positive/negative/stable)
     */
    fun calculateSleepTrend(days: Int): String {
        val records = sleepTrackerService.getAllRecords()
        if (records.size < 2) return "Insufficient data"

        val sevenDaysAgo = System.currentTimeMillis() - (7 * 24 * 60 * 60 * 1000)
        val recentRecords = records.filter { it.startTime >= sevenDaysAgo }

        if (recentRecords.size < 2) return "Insufficient data"

        val goal = sleepTrackerService.getBedtimeGoal()
        val recentQuality = recentRecords.map { calculateQualityScore(it.duration, goal) }.average()
        val earlierRecords = records.filter { it.startTime < sevenDaysAgo }
        val earlierQuality = if (earlierRecords.isNotEmpty()) {
            earlierRecords.map { calculateQualityScore(it.duration, goal) }.average()
        } else {
            recentQuality
        }

        return when {
            recentQuality > earlierQuality + 5 -> "↑ Improving"
            recentQuality < earlierQuality - 5 -> "↓ Declining"
            else -> "→ Stable"
        }
    }

    /**
     * Private helper functions
     */

    private fun getQualityForDays(records: List<SleepRecord>, days: Int): List<QualityDataPoint> {
        val goal = sleepTrackerService.getBedtimeGoal()
        val dateFormat = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault())
        val qualityMap = mutableMapOf<String, Float>()

        records.forEach { record ->
            val quality = calculateQualityScore(record.duration, goal).toFloat()
            qualityMap[record.date] = quality
        }

        // Fill in last N days
        val result = mutableListOf<QualityDataPoint>()
        val calendar = Calendar.getInstance()
        calendar.add(Calendar.DAY_OF_YEAR, -days)

        repeat(days) {
            val dateStr = dateFormat.format(calendar.time)
            val quality = qualityMap[dateStr] ?: 0f
            result.add(QualityDataPoint(dateStr, quality))
            calendar.add(Calendar.DAY_OF_YEAR, 1)
        }

        return result
    }

    private fun getQualityForWeeks(records: List<SleepRecord>, weeks: Int): List<QualityDataPoint> {
        val goal = sleepTrackerService.getBedtimeGoal()
        val result = mutableListOf<QualityDataPoint>()
        val calendar = Calendar.getInstance()
        calendar.add(Calendar.WEEK_OF_YEAR, -weeks)

        repeat(weeks) {
            val weekStart = calendar.time
            calendar.add(Calendar.DAY_OF_YEAR, 7)
            val weekEnd = calendar.time

            val weekRecords = records.filter { record ->
                try {
                    val date = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault()).parse(record.date)
                    date != null && date >= weekStart && date <= weekEnd
                } catch (e: Exception) {
                    false
                }
            }

            val avgQuality = if (weekRecords.isNotEmpty()) {
                weekRecords.map { calculateQualityScore(it.duration, goal).toFloat() }.average()
            } else {
                0f
            }

            result.add(QualityDataPoint(SimpleDateFormat("yyyy-MM-dd", Locale.getDefault()).format(weekEnd), avgQuality.toFloat()))
        }

        return result
    }

    private fun getQualityForMonths(records: List<SleepRecord>, months: Int): List<QualityDataPoint> {
        val goal = sleepTrackerService.getBedtimeGoal()
        val result = mutableListOf<QualityDataPoint>()
        val calendar = Calendar.getInstance()
        calendar.add(Calendar.MONTH, -months)

        repeat(months) {
            val monthStart = calendar.time
            val monthEnd = Calendar.getInstance().apply {
                time = monthStart
                set(Calendar.DAY_OF_MONTH, getActualMaximum(Calendar.DAY_OF_MONTH))
            }.time

            val monthRecords = records.filter { record ->
                try {
                    val date = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault()).parse(record.date)
                    date != null && date >= monthStart && date <= monthEnd
                } catch (e: Exception) {
                    false
                }
            }

            val avgQuality = if (monthRecords.isNotEmpty()) {
                monthRecords.map { calculateQualityScore(it.duration, goal).toFloat() }.average()
            } else {
                0f
            }

            result.add(QualityDataPoint(SimpleDateFormat("yyyy-MM", Locale.getDefault()).format(monthEnd), avgQuality.toFloat()))
            calendar.add(Calendar.MONTH, 1)
        }

        return result
    }

    private fun getQualityForAllTime(records: List<SleepRecord>): List<QualityDataPoint> {
        val goal = sleepTrackerService.getBedtimeGoal()
        val monthMap = mutableMapOf<String, MutableList<Float>>()

        records.forEach { record ->
            val monthKey = record.date.substring(0, 7) // yyyy-MM
            val quality = calculateQualityScore(record.duration, goal).toFloat()
            monthMap.getOrPut(monthKey) { mutableListOf() }.add(quality)
        }

        return monthMap.map { (month, qualities) ->
            QualityDataPoint(month, qualities.average().toFloat())
        }.sortedBy { it.date }
    }

    private fun formatDuration(ms: Long): String {
        val hours = ms / (1000 * 60 * 60)
        val minutes = (ms % (1000 * 60 * 60)) / (1000 * 60)
        return "${hours}h ${minutes}m"
    }
}
