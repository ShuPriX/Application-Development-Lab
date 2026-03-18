package com.example.sleeptracker.data

import org.json.JSONObject

data class SleepRecord(
    val startTime: Long,
    val endTime: Long,
    val duration: Long,
    val sleepStage: String,
    val date: String, // YYYY-MM-DD format
    val qualityScore: Int = 0, // 0-100
    val notes: String = ""
) {
    fun toJson(): JSONObject {
        return JSONObject().apply {
            put("startTime", startTime)
            put("endTime", endTime)
            put("duration", duration)
            put("sleepStage", sleepStage)
            put("date", date)
            put("qualityScore", qualityScore)
            put("notes", notes)
        }
    }

    companion object {
        fun fromJson(json: JSONObject): SleepRecord {
            return SleepRecord(
                startTime = json.getLong("startTime"),
                endTime = json.getLong("endTime"),
                duration = json.getLong("duration"),
                sleepStage = json.getString("sleepStage"),
                date = json.getString("date"),
                qualityScore = json.optInt("qualityScore", 0),
                notes = json.optString("notes", "")
            )
        }
    }
}
