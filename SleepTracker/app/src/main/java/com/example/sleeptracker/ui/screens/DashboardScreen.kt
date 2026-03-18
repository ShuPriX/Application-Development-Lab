package com.example.sleeptracker.ui.screens

import android.content.Context
import android.graphics.Color as AndroidColor
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import com.example.sleeptracker.data.AnalyticsService
import com.example.sleeptracker.data.SleepTrackerService
import com.example.sleeptracker.ui.theme.*
import com.example.sleeptracker.utils.ChartFormatter
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.data.LineData

@Composable
fun DashboardScreen(
    context: Context,
    onTrackingClick: () -> Unit
) {
    val sleepTrackerService = remember { SleepTrackerService(context) }
    val analyticsService = remember { AnalyticsService(sleepTrackerService) }

    var selectedPeriod: AnalyticsService.TimePeriod by remember { mutableStateOf(AnalyticsService.TimePeriod.Days) }
    var qualityData by remember { mutableStateOf(analyticsService.getQualityScoresForPeriod(selectedPeriod)) }
    var quickStats by remember { mutableStateOf(analyticsService.getQuickStats()) }
    var lastSevenNights by remember { mutableStateOf(analyticsService.getLastSevenNightsData()) }
    var trend by remember { mutableStateOf(analyticsService.calculateSleepTrend(7)) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(DarkBackground)
            .verticalScroll(rememberScrollState())
            .padding(16.dp)
            .systemBarsPadding(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Header
        Text(
            text = "Sleep Quality",
            fontSize = 28.sp,
            fontWeight = FontWeight.Bold,
            color = LightText,
            modifier = Modifier.align(Alignment.Start)
        )

        // Sleep Quality Chart
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(280.dp)
                .background(DarkSurface, RoundedCornerShape(16.dp))
                .padding(12.dp)
        ) {
            if (qualityData.isNotEmpty()) {
                ChartView(qualityData = qualityData, period = selectedPeriod)
            } else {
                Box(
                    modifier = Modifier
                        .fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "No sleep data available",
                        color = LightText.copy(alpha = 0.6f),
                        fontSize = 14.sp
                    )
                }
            }
        }

        // Time Period Filters
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .horizontalScroll(rememberScrollState()),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            listOf(
                "Days" to AnalyticsService.TimePeriod.Days,
                "Weeks" to AnalyticsService.TimePeriod.Weeks,
                "Months" to AnalyticsService.TimePeriod.Months,
                "All" to AnalyticsService.TimePeriod.All
            ).forEach { (label, period) ->
                FilterButton(
                    text = label,
                    isSelected = selectedPeriod == period,
                    onClick = {
                        selectedPeriod = period
                        qualityData = analyticsService.getQualityScoresForPeriod(period)
                    }
                )
            }
        }

        // Quick Stats
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(DarkSurface, RoundedCornerShape(16.dp))
                .padding(16.dp)
        ) {
            Column(
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Text(
                    text = "Summary",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = LightText
                )

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    StatItem(
                        label = "Best Night",
                        value = quickStats.bestNight,
                        modifier = Modifier.weight(1f)
                    )
                    StatItem(
                        label = "Worst Night",
                        value = quickStats.worstNight,
                        modifier = Modifier.weight(1f)
                    )
                }

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    StatItem(
                        label = "Total Nights",
                        value = quickStats.totalNights.toString(),
                        modifier = Modifier.weight(1f)
                    )
                    StatItem(
                        label = "Avg Quality",
                        value = "${quickStats.avgQuality.toInt()}%",
                        modifier = Modifier.weight(1f)
                    )
                }

                Text(
                    text = "Trend: $trend",
                    fontSize = 14.sp,
                    color = AccentBlue,
                    fontWeight = FontWeight.Medium
                )
            }
        }

        // Last 7 Nights Preview
        if (lastSevenNights.isNotEmpty()) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(DarkSurface, RoundedCornerShape(16.dp))
                    .padding(16.dp)
            ) {
                Column(
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Text(
                        text = "Last 7 Nights",
                        fontSize = 16.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = LightText
                    )

                    Column(
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        lastSevenNights.forEachIndexed { index, comparison ->
                            if (index < 7) {
                                ComparisonRow(
                                    date = comparison.date,
                                    actualSleep = comparison.actualSleep,
                                    goal = comparison.goal,
                                    metGoal = comparison.metGoal
                                )
                            }
                        }
                    }
                }
            }
        }

        // Quick Actions
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(DarkSurface, RoundedCornerShape(16.dp))
                .padding(16.dp)
        ) {
            Column(
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Text(
                    text = "Quick Actions",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = LightText
                )

                Row(
                    modifier = Modifier
                        .fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    QuickActionButton(
                        icon = "🧘",
                        label = "Meditation",
                        modifier = Modifier.weight(1f),
                        onClick = {}
                    )
                    QuickActionButton(
                        icon = "😴",
                        label = "Sleep",
                        modifier = Modifier.weight(1f),
                        onClick = onTrackingClick
                    )
                    QuickActionButton(
                        icon = "📔",
                        label = "Journal",
                        modifier = Modifier.weight(1f),
                        onClick = {}
                    )
                }

                Row(
                    modifier = Modifier
                        .fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    QuickActionButton(
                        icon = "📊",
                        label = "Stats",
                        modifier = Modifier.weight(1f),
                        onClick = {}
                    )
                    QuickActionButton(
                        icon = "👤",
                        label = "Profile",
                        modifier = Modifier.weight(1f),
                        onClick = {}
                    )
                    Spacer(modifier = Modifier.weight(1f))
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))
    }
}

@Composable
fun ChartView(qualityData: List<com.example.sleeptracker.data.QualityDataPoint>, period: AnalyticsService.TimePeriod) {
    AndroidView(
        modifier = Modifier.fillMaxSize(),
        factory = { context ->
            LineChart(context).apply {
                description.isEnabled = false
                legend.isEnabled = true
                legend.textColor = AndroidColor.WHITE
                xAxis.apply {
                    textColor = AndroidColor.WHITE
                    position = com.github.mikephil.charting.components.XAxis.XAxisPosition.BOTTOM
                    setDrawGridLines(false)
                }
                axisLeft.apply {
                    textColor = AndroidColor.WHITE
                    axisMinimum = 0f
                    axisMaximum = 100f
                    setDrawGridLines(true)
                }
                axisRight.apply {
                    isEnabled = false
                }
            }
        },
        update = { chart ->
            val dataSets = mutableListOf(
                ChartFormatter.convertToLineDataSet(
                    qualityData,
                    "Sleep Quality",
                    AccentBlue.toArgb()
                )
            )

            val goalEntries = ChartFormatter.getGoalLineEntries(qualityData.size)
            if (goalEntries.isNotEmpty()) {
                dataSets.add(
                    com.github.mikephil.charting.data.LineDataSet(goalEntries, "Goal").apply {
                        color = GoalGreen.toArgb()
                        setCircleColor(GoalGreen.toArgb())
                        lineWidth = 2f
                        circleRadius = 3f
                        setDrawFilled(false)
                        mode = com.github.mikephil.charting.data.LineDataSet.Mode.LINEAR
                    }
                )
            }

            chart.data = LineData(dataSets as List<com.github.mikephil.charting.interfaces.datasets.ILineDataSet>)
            chart.xAxis.valueFormatter = com.github.mikephil.charting.formatter.IndexAxisValueFormatter(
                ChartFormatter.getAbbreviatedDateLabels(qualityData, when (period) {
                    AnalyticsService.TimePeriod.Days -> "Days"
                    AnalyticsService.TimePeriod.Weeks -> "Weeks"
                    AnalyticsService.TimePeriod.Months -> "Months"
                    AnalyticsService.TimePeriod.All -> "All"
                })
            )
            chart.invalidate()
        }
    )
}

@Composable
fun FilterButton(
    text: String,
    isSelected: Boolean,
    onClick: () -> Unit
) {
    Button(
        onClick = onClick,
        modifier = Modifier
            .height(36.dp)
            .then(
                if (!isSelected) Modifier.border(
                    width = 1.dp,
                    color = AccentBlue.copy(alpha = 0.5f),
                    shape = RoundedCornerShape(20.dp)
                ) else Modifier
            ),
        shape = RoundedCornerShape(20.dp),
        colors = ButtonDefaults.buttonColors(
            containerColor = if (isSelected) AccentBlue else DarkBackground,
            contentColor = LightText
        )
    ) {
        Text(text = text, fontSize = 12.sp, fontWeight = FontWeight.Medium)
    }
}

@Composable
fun StatItem(
    label: String,
    value: String,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .background(DarkBackground, RoundedCornerShape(8.dp))
            .padding(12.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(text = label, fontSize = 11.sp, color = LightText.copy(alpha = 0.7f))
        Spacer(modifier = Modifier.height(4.dp))
        Text(text = value, fontSize = 14.sp, fontWeight = FontWeight.SemiBold, color = AccentBlue)
    }
}

@Composable
fun ComparisonRow(
    date: String,
    actualSleep: String,
    goal: String,
    metGoal: Boolean
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(DarkBackground, RoundedCornerShape(8.dp))
            .padding(12.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(text = date, fontSize = 12.sp, color = LightText, fontWeight = FontWeight.Medium)
        Text(
            text = actualSleep,
            fontSize = 12.sp,
            color = if (metGoal) GoalGreen else AwakeColor,
            fontWeight = FontWeight.SemiBold
        )
    }
}

@Composable
fun QuickActionButton(
    icon: String,
    label: String,
    modifier: Modifier = Modifier,
    onClick: () -> Unit
) {
    Button(
        onClick = onClick,
        modifier = modifier
            .height(80.dp),
        shape = RoundedCornerShape(12.dp),
        colors = ButtonDefaults.buttonColors(
            containerColor = AccentBlue.copy(alpha = 0.2f)
        )
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Text(text = icon, fontSize = 24.sp)
            Spacer(modifier = Modifier.height(4.dp))
            Text(text = label, fontSize = 10.sp, color = LightText)
        }
    }
}
