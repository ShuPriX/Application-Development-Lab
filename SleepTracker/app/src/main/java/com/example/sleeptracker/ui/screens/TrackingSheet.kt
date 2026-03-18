package com.example.sleeptracker.ui.screens

import android.content.Context
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.sleeptracker.data.SleepRecord
import com.example.sleeptracker.data.SleepTrackerService
import com.example.sleeptracker.ui.theme.*
import kotlinx.coroutines.delay
import java.text.SimpleDateFormat
import java.util.*

@Composable
fun TrackingSheet(
    context: Context,
    onDismiss: () -> Unit
) {
    val sleepTracker = remember { SleepTrackerService(context) }
    var isTracking by remember { mutableStateOf(false) }
    var currentDuration by remember { mutableStateOf(0L) }
    var startTime by remember { mutableStateOf(0L) }
    var sleepStage by remember { mutableStateOf("Ready to Track") }
    var averageSleep by remember { mutableStateOf("0h 0m") }

    // Update timer
    LaunchedEffect(isTracking) {
        while (isTracking) {
            delay(1000)
            val elapsed = System.currentTimeMillis() - startTime
            currentDuration = elapsed
            sleepStage = sleepTracker.getSleepStage(elapsed)
        }
    }

    // Update average sleep display
    LaunchedEffect(Unit) {
        averageSleep = sleepTracker.calculateAverageSleep()
    }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(DarkBackground)
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Handle bar
        Box(
            modifier = Modifier
                .width(40.dp)
                .height(4.dp)
                .background(LightText.copy(alpha = 0.3f), RoundedCornerShape(2.dp))
        )

        // Header
        Text(
            text = "Sleep Tracking",
            fontSize = 24.sp,
            fontWeight = FontWeight.Bold,
            color = LightText
        )

        // Timer Display
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(DarkSurface, RoundedCornerShape(16.dp))
                .padding(32.dp),
            contentAlignment = Alignment.Center
        ) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {
                Text(
                    text = formatTimeShort(currentDuration),
                    fontSize = 56.sp,
                    fontWeight = FontWeight.Bold,
                    color = AccentBlue
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Elapsed Time",
                    fontSize = 14.sp,
                    color = LightText.copy(alpha = 0.7f)
                )
            }
        }

        // Sleep Stage Display
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(DarkSurface, RoundedCornerShape(16.dp))
                .padding(20.dp),
            contentAlignment = Alignment.Center
        ) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = sleepStage,
                    fontSize = 20.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = when (sleepStage) {
                        "Deep Sleep" -> DeepSleepColor
                        "Light Sleep" -> LightSleepColor
                        "Awake" -> AwakeColor
                        else -> AccentBlue
                    }
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = if (isTracking) "Currently tracking" else "Not tracking",
                    fontSize = 12.sp,
                    color = LightText.copy(alpha = 0.6f)
                )
            }
        }

        // Control Buttons
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .height(60.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            SleepTrackerButton(
                text = "Start",
                backgroundColor = if (!isTracking) GoalGreen else GoalGreen.copy(alpha = 0.5f),
                enabled = !isTracking,
                modifier = Modifier.weight(1f),
                onClick = {
                    if (!isTracking) {
                        startTime = System.currentTimeMillis()
                        currentDuration = 0L
                        sleepStage = "Awake"
                        isTracking = true
                    }
                }
            )

            SleepTrackerButton(
                text = "Stop",
                backgroundColor = if (isTracking) AwakeColor else AwakeColor.copy(alpha = 0.5f),
                enabled = isTracking,
                modifier = Modifier.weight(1f),
                onClick = {
                    if (isTracking) {
                        isTracking = false
                        // Save record
                        val record = SleepRecord(
                            startTime = startTime,
                            endTime = System.currentTimeMillis(),
                            duration = currentDuration,
                            sleepStage = sleepStage,
                            date = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault()).format(Date()),
                            qualityScore = sleepTracker.calculateQualityScore(
                                currentDuration,
                                sleepTracker.getBedtimeGoal()
                            )
                        )
                        sleepTracker.saveSleepRecord(record)
                        averageSleep = sleepTracker.calculateAverageSleep()
                    }
                }
            )

            SleepTrackerButton(
                text = "Reset",
                backgroundColor = AccentBlue,
                modifier = Modifier.weight(1f),
                onClick = {
                    isTracking = false
                    currentDuration = 0L
                    sleepStage = "Ready to Track"
                }
            )
        }

        // Close Button
        SleepTrackerButton(
            text = "Close",
            backgroundColor = DarkSurface,
            modifier = Modifier.fillMaxWidth(),
            onClick = onDismiss
        )
    }
}

@Composable
fun SleepTrackerButton(
    text: String,
    backgroundColor: Color,
    enabled: Boolean = true,
    modifier: Modifier = Modifier,
    onClick: () -> Unit
) {
    Button(
        onClick = onClick,
        modifier = modifier
            .height(56.dp),
        shape = RoundedCornerShape(12.dp),
        colors = ButtonDefaults.buttonColors(
            containerColor = backgroundColor,
            disabledContainerColor = backgroundColor.copy(alpha = 0.5f)
        ),
        enabled = enabled
    ) {
        Text(
            text = text,
            fontSize = 16.sp,
            fontWeight = FontWeight.SemiBold,
            color = Color.White
        )
    }
}

fun formatTimeShort(ms: Long): String {
    val hours = ms / (1000 * 60 * 60)
    val minutes = (ms % (1000 * 60 * 60)) / (1000 * 60)
    val seconds = (ms % (1000 * 60)) / 1000
    return String.format("%02d:%02d:%02d", hours, minutes, seconds)
}

/**
 * Extension function to calculate quality score for a SleepTrackerService
 */
fun SleepTrackerService.calculateQualityScore(duration: Long, goalMs: Long): Int {
    val percentage = ((duration.toFloat() / goalMs.toFloat()) * 100).toInt()
    return minOf(100, maxOf(0, percentage))
}
