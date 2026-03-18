package com.example.sleeptracker.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.sleeptracker.data.SleepTrackerService
import com.example.sleeptracker.ui.theme.*

@Composable
fun SettingsScreen(
    serviceParam: android.content.Context,
    sleepTracker: SleepTrackerService,
    onBackClick: () -> Unit
) {
    var goalHours by remember { mutableStateOf((sleepTracker.getBedtimeGoal() / (1000 * 60 * 60)).toInt()) }
    var goalMinutes by remember { mutableStateOf(((sleepTracker.getBedtimeGoal() % (1000 * 60 * 60)) / (1000 * 60)).toInt()) }
    var showAlert by remember { mutableStateOf(false) }
    var alertMessage by remember { mutableStateOf("") }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(DarkBackground)
            .padding(16.dp)
            .systemBarsPadding()
            .verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Header
        Text(
            text = "Settings",
            fontSize = 32.sp,
            fontWeight = FontWeight.Bold,
            color = LightText,
            modifier = Modifier.padding(top = 16.dp)
        )

        Spacer(modifier = Modifier.height(8.dp))

        // Bedtime Goal Card
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(DarkSurface, RoundedCornerShape(16.dp))
                .padding(20.dp)
        ) {
            Column(
                modifier = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Text(
                    text = "Bedtime Goal",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = LightText
                )

                // Hours Picker
                Row(
                    modifier = Modifier
                        .fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Text(
                        text = "Hours:",
                        fontSize = 14.sp,
                        color = LightText,
                        modifier = Modifier.width(60.dp)
                    )
                    TextField(
                        value = goalHours.toString(),
                        onValueChange = { newValue ->
                            newValue.toIntOrNull()?.let { value ->
                                if (value in 0..23) goalHours = value
                            }
                        },
                        modifier = Modifier
                            .weight(1f)
                            .height(48.dp),
                        shape = RoundedCornerShape(8.dp),
                        colors = TextFieldDefaults.colors(
                            focusedContainerColor = DarkBackground,
                            unfocusedContainerColor = DarkBackground,
                            focusedTextColor = AccentBlue,
                            unfocusedTextColor = LightText
                        ),
                        singleLine = true
                    )
                }

                // Minutes Picker
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Text(
                        text = "Minutes:",
                        fontSize = 14.sp,
                        color = LightText,
                        modifier = Modifier.width(60.dp)
                    )
                    TextField(
                        value = goalMinutes.toString(),
                        onValueChange = { newValue ->
                            newValue.toIntOrNull()?.let { value ->
                                if (value in 0..59) goalMinutes = value
                            }
                        },
                        modifier = Modifier
                            .weight(1f)
                            .height(48.dp),
                        shape = RoundedCornerShape(8.dp),
                        colors = TextFieldDefaults.colors(
                            focusedContainerColor = DarkBackground,
                            unfocusedContainerColor = DarkBackground,
                            focusedTextColor = AccentBlue,
                            unfocusedTextColor = LightText
                        ),
                        singleLine = true
                    )
                }

                Text(
                    text = "Goal: ${goalHours}h ${goalMinutes}m per day",
                    fontSize = 14.sp,
                    color = GoalGreen,
                    fontWeight = FontWeight.Medium
                )
            }
        }

        // Today's Comparison
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(DarkSurface, RoundedCornerShape(16.dp))
                .padding(20.dp)
        ) {
            Column(
                modifier = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text(
                    text = "Today's Progress",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = LightText
                )
                Text(
                    text = sleepTracker.getTodayComparisonWithGoal(),
                    fontSize = 16.sp,
                    color = AccentBlue,
                    fontWeight = FontWeight.Medium
                )
            }
        }

        Spacer(modifier = Modifier.weight(1f))

        // Save Button
        SleepTrackerButton(
            text = "Save Goal",
            backgroundColor = GoalGreen,
            modifier = Modifier.fillMaxWidth()
        ) {
            val goalMs = (goalHours * 60 * 60 * 1000L) + (goalMinutes * 60 * 1000L)
            if (goalMs > 0) {
                sleepTracker.setBedtimeGoal(goalMs)
                alertMessage = "Goal saved successfully!"
                showAlert = true
            } else {
                alertMessage = "Please enter a valid goal time"
                showAlert = true
            }
        }

        // Back Button
        SleepTrackerButton(
            text = "Back",
            backgroundColor = AccentBlue,
            modifier = Modifier.fillMaxWidth()
        ) {
            onBackClick()
        }
    }

    if (showAlert) {
        AlertDialog(
            onDismissRequest = { showAlert = false },
            title = {
                Text(
                    "Message",
                    color = LightText,
                    fontWeight = FontWeight.SemiBold
                )
            },
            text = {
                Text(
                    alertMessage,
                    color = LightText
                )
            },
            confirmButton = {
                Button(
                    onClick = { showAlert = false },
                    colors = ButtonDefaults.buttonColors(
                        containerColor = AccentBlue
                    )
                ) {
                    Text("OK", color = DarkBackground)
                }
            },
            containerColor = DarkSurface,
            textContentColor = LightText
        )
    }
}
