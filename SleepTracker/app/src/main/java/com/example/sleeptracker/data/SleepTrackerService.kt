package com.example.sleeptracker.data

import android.content.Context
import android.content.SharedPreferences
import org.json.JSONArray
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class SleepTrackerService(context: Context) {
    private val sharedPreferences: SharedPreferences =
        context.getSharedPreferences("sleep_tracker", Context.MODE_PRIVATE)

    private companion object {
        const val RECORDS_KEY = "sleep_records"
        const val GOAL_KEY = "bedtime_goal"
    }

    fun saveSleepRecord(record: SleepRecord) {
        val records = getAllRecords().toMutableList()
        records.add(record)

        val jsonArray = JSONArray()
        records.forEach { jsonArray.put(it.toJson()) }

        sharedPreferences.edit().putString(RECORDS_KEY, jsonArray.toString()).apply()
    }

    fun getAllRecords(): List<SleepRecord> {
        val recordsJson = sharedPreferences.getString(RECORDS_KEY, "[]") ?: "[]"
        val jsonArray = JSONArray(recordsJson)
        val records = mutableListOf<SleepRecord>()

        for (i in 0 until jsonArray.length()) {
            records.add(SleepRecord.fromJson(jsonArray.getJSONObject(i)))
        }
        return records
    }

    fun getTodayRecords(): List<SleepRecord> {
        val today = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault()).format(Date())
        return getAllRecords().filter { it.date == today }
    }

    fun calculateAverageSleep(): String {
        val records = getAllRecords()
        if (records.isEmpty()) return "0h 0m"

        val totalDuration = records.sumOf { it.duration }
        val hours = totalDuration / (1000 * 60 * 60)
        val minutes = (totalDuration % (1000 * 60 * 60)) / (1000 * 60)

        return "${hours}h ${minutes}m"
    }

    fun getLastSevenDaysAverage(): String {
        val records = getAllRecords()
        if (records.isEmpty()) return "0h 0m"

        val sevenDaysAgo = System.currentTimeMillis() - (7 * 24 * 60 * 60 * 1000)
        val recentRecords = records.filter { it.startTime >= sevenDaysAgo }

        if (recentRecords.isEmpty()) return "0h 0m"

        val totalDuration = recentRecords.sumOf { it.duration }
        val hours = totalDuration / (1000 * 60 * 60)
        val minutes = (totalDuration % (1000 * 60 * 60)) / (1000 * 60)

        return "${hours}h ${minutes}m"
    }

    fun getBedtimeGoal(): Long {
        return sharedPreferences.getLong(GOAL_KEY, 8 * 60 * 60 * 1000) // Default 8 hours
    }

    fun setBedtimeGoal(goalMs: Long) {
        sharedPreferences.edit().putLong(GOAL_KEY, goalMs).apply()
    }

    fun clearAllData() {
        sharedPreferences.edit().clear().apply()
    }

    fun formatDuration(durationMs: Long): String {
        val hours = durationMs / (1000 * 60 * 60)
        val minutes = (durationMs % (1000 * 60 * 60)) / (1000 * 60)
        val seconds = (durationMs % (1000 * 60)) / 1000
        return String.format("%02d:%02d:%02d", hours, minutes, seconds)
    }

    fun getSleepStage(durationMs: Long): String {
        return when {
            durationMs >= 90 * 60 * 1000 -> "Deep Sleep"
            durationMs >= 30 * 60 * 1000 -> "Light Sleep"
            else -> "Awake"
        }
    }

    fun getTodayComparisonWithGoal(): String {
        val todayRecords = getTodayRecords()
        val totalToday = todayRecords.sumOf { it.duration }
        val goal = getBedtimeGoal()

        return if (totalToday >= goal) {
            val extra = totalToday - goal
            val hours = extra / (1000 * 60 * 60)
            val minutes = (extra % (1000 * 60 * 60)) / (1000 * 60)
            "You slept ${hours}h ${minutes}m more than your goal!"
        } else {
            val less = goal - totalToday
            val hours = less / (1000 * 60 * 60)
            val minutes = (less % (1000 * 60 * 60)) / (1000 * 60)
            "You slept ${hours}h ${minutes}m less than your goal"
        }
    }

    fun calculateQualityScore(duration: Long, goalMs: Long): Int {
        val percentage = ((duration.toFloat() / goalMs.toFloat()) * 100).toInt()
        return minOf(100, maxOf(0, percentage))
    }
}
