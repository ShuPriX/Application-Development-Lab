package com.example.sleeptracker.utils

import com.example.sleeptracker.data.QualityDataPoint
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineDataSet

object ChartFormatter {

    /**
     * Convert QualityDataPoints to LineDataSet for MPAndroidChart
     */
    fun convertToLineDataSet(
        dataPoints: List<QualityDataPoint>,
        label: String,
        color: Int
    ): LineDataSet {
        val entries = dataPoints.mapIndexed { index, point ->
            Entry(index.toFloat(), point.quality)
        }

        return LineDataSet(entries, label).apply {
            this.color = color
            this.setCircleColor(color)
            this.lineWidth = 2.5f
            this.circleRadius = 4f
            this.highLightColor = color
            this.setDrawFilled(true)
            this.fillAlpha = 100
            this.fillColor = color
            this.mode = LineDataSet.Mode.CUBIC_BEZIER
        }
    }

    /**
     * Format date labels for chart X-axis
     */
    fun getDateLabels(dataPoints: List<QualityDataPoint>): List<String> {
        return dataPoints.map { it.date }
    }

    /**
     * Get abbreviated date labels for cleaner chart display
     */
    fun getAbbreviatedDateLabels(dataPoints: List<QualityDataPoint>, period: String): List<String> {
        return when (period) {
            "Days" -> dataPoints.map { it.date.substring(5) } // MM-DD
            "Weeks" -> dataPoints.map { it.date.substring(5) } // MM-DD
            "Months" -> dataPoints.map { it.date.substring(0, 7) } // yyyy-MM
            else -> dataPoints.map { it.date }
        }
    }

    /**
     * Get Y-axis minimum and maximum for proper scaling
     */
    fun getYAxisBounds(dataPoints: List<QualityDataPoint>): Pair<Float, Float> {
        val values = dataPoints.map { it.quality }
        if (values.isEmpty()) return Pair(0f, 100f)

        val min = values.minOrNull() ?: 0f
        val max = values.maxOrNull() ?: 100f

        // Add padding
        val range = max - min
        val padding = if (range == 0f) 50f else range * 0.1f

        return Pair(
            maxOf(0f, min - padding),
            minOf(100f, max + padding)
        )
    }

    /**
     * Get goal line value (constant at 100%)
     */
    fun getGoalLineEntries(count: Int): List<Entry> {
        return (0 until count).map { Entry(it.toFloat(), 100f) }
    }

    /**
     * Format percentage value for display
     */
    fun formatQualityPercentage(quality: Float): String {
        return "${quality.toInt()}%"
    }

    /**
     * Get color based on quality score
     */
    fun getQualityColor(quality: Float): Int {
        return when {
            quality >= 90 -> 0xFF66BB6A.toInt() // Green
            quality >= 75 -> 0xFF42A5F5.toInt() // Light Blue
            quality >= 60 -> 0xFFFFA726.toInt() // Orange
            else -> 0xFFEF5350.toInt() // Red
        }
    }
}
