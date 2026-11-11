Attribute VB_Name = "SECOM_SPC"
Option Explicit
' SPC macros for the SECOM workbook. Works on Excel for Windows and Mac.
' Sheets: Control (inputs and results), Sensors (run_id, timestamp, fail, S### columns), Chart (written by macros).

Private Const FIRST_ROW As Long = 2

Private Function SensorCol(ByVal nm As String) As Long
    Dim c As Long
    For c = 4 To Worksheets("Sensors").Cells(1, Columns.Count).End(xlToLeft).Column
        If Worksheets("Sensors").Cells(1, c).Value = nm Then SensorCol = c: Exit Function
    Next c
    SensorCol = 0
End Function

' Individuals chart limits from rows 2..lastRow of the Sensors sheet: mean +/- k * MRbar / 1.128
Private Sub Limits(ByVal col As Long, ByVal r1 As Long, ByVal r2 As Long, ByVal k As Double, _
                   ByRef cl As Double, ByRef lcl As Double, ByRef ucl As Double, ByRef mrbar As Double)
    Dim r As Long, n As Long, s As Double, m As Long, mr As Double, prev As Variant, v As Variant
    prev = Empty
    For r = r1 To r2
        v = Worksheets("Sensors").Cells(r, col).Value
        If IsNumeric(v) And Not IsEmpty(v) Then
            n = n + 1: s = s + v
            If Not IsEmpty(prev) Then mr = mr + Abs(v - prev): m = m + 1
            prev = v
        End If
    Next r
    If n < 2 Or m < 1 Then Err.Raise vbObjectError + 1, , "Not enough numeric values to set limits."
    cl = s / n: mrbar = mr / m
    ucl = cl + k * mrbar / 1.128: lcl = cl - k * mrbar / 1.128
End Sub

Public Sub RunSensorChart()
    Dim wsC As Worksheet, wsS As Worksheet, wsO As Worksheet
    Set wsC = Worksheets("Control"): Set wsS = Worksheets("Sensors")
    Dim nm As String, col As Long, p1 As Long, k As Double
    nm = wsC.Range("B4").Value: p1 = wsC.Range("B5").Value: k = wsC.Range("B7").Value
    col = SensorCol(nm)
    If col = 0 Then MsgBox "Sensor " & nm & " not found.": Exit Sub
    Dim last As Long: last = wsS.Cells(Rows.Count, 1).End(xlUp).Row
    Dim cl As Double, lcl As Double, ucl As Double, mrbar As Double
    Limits col, FIRST_ROW, p1 + 1, k, cl, lcl, ucl, mrbar

    Application.DisplayAlerts = False
    On Error Resume Next
    Worksheets("Chart").Delete
    On Error GoTo 0
    Application.DisplayAlerts = True
    Set wsO = Worksheets.Add(After:=Worksheets(Worksheets.Count)): wsO.Name = "Chart"
    wsO.Range("A1:G1").Value = Array("run_id", "timestamp", "value", "CL", "LCL", "UCL", "nelson_rule")

    Dim r As Long, o As Long, v As Variant, f As String
    Dim side As Long, runSame As Long, lastSide As Long, up As Long, dn As Long, prevV As Double, havePrev As Boolean
    Dim flags As Long, alarmDays As Long, lastAlarmDay As Double, sig As Double
    sig = (ucl - cl) / k
    For r = FIRST_ROW To last
        v = wsS.Cells(r, col).Value
        o = r
        wsO.Cells(o, 1).Value = wsS.Cells(r, 1).Value
        wsO.Cells(o, 2).Value = wsS.Cells(r, 2).Value
        wsO.Cells(o, 4).Value = cl: wsO.Cells(o, 5).Value = lcl: wsO.Cells(o, 6).Value = ucl
        f = ""
        If IsNumeric(v) And Not IsEmpty(v) Then
            wsO.Cells(o, 3).Value = v
            If v > ucl Or v < lcl Then f = f & "1 "
            ' rule 2: 9 in a row on one side of CL
            If v > cl Then
                side = 1
            ElseIf v < cl Then
                side = -1
            Else
                side = 0
            End If
            If side <> 0 And side = lastSide Then
                runSame = runSame + 1
            ElseIf side = 0 Then
                runSame = 0
            Else
                runSame = 1
            End If
            lastSide = side
            If runSame >= 9 Then f = f & "2 "
            ' rule 3: 6 points steadily rising or falling
            If havePrev Then
                If v > prevV Then
                    up = up + 1: dn = 0
                ElseIf v < prevV Then
                    dn = dn + 1: up = 0
                Else
                    up = 0: dn = 0
                End If
            End If
            If up >= 5 Or dn >= 5 Then f = f & "3 "
            havePrev = True: prevV = v
        End If
        wsO.Cells(o, 7).Value = Trim(f)
        If Len(f) > 0 Then
            flags = flags + 1
            If InStr(f, "1") > 0 Then
                If Int(wsS.Cells(r, 2).Value) <> lastAlarmDay Then alarmDays = alarmDays + 1: lastAlarmDay = Int(wsS.Cells(r, 2).Value)
            End If
        End If
    Next r
    wsC.Range("B10").Value = cl: wsC.Range("B11").Value = lcl: wsC.Range("B12").Value = ucl
    wsC.Range("B13").Value = mrbar: wsC.Range("B14").Value = flags: wsC.Range("B15").Value = alarmDays
    wsO.Range("B:B").NumberFormat = "yyyy-mm-dd hh:mm": wsO.Columns("A:G").AutoFit

    Dim co As ChartObject
    Set co = wsO.ChartObjects.Add(Left:=wsO.Range("I2").Left, Top:=wsO.Range("I2").Top, Width:=620, Height:=320)
    co.Chart.SetSourceData Source:=wsO.Range(wsO.Cells(1, 3), wsO.Cells(last, 6)), PlotBy:=xlColumns
    co.Chart.ChartType = xlLine
    co.Chart.HasTitle = True
    co.Chart.ChartTitle.Text = nm & " individuals chart, limits from Phase I (runs 1-" & p1 & ")"
    co.Chart.HasLegend = True
    wsC.Activate
    MsgBox nm & ": " & flags & " flagged runs on " & alarmDays & " days (rule 1 days).", vbInformation, "SECOM SPC"
End Sub

' Compare frozen Phase I limits with limits re-baselined on the 200 runs after the change point.
Public Sub RunRebaseline()
    Dim wsC As Worksheet, wsS As Worksheet
    Set wsC = Worksheets("Control"): Set wsS = Worksheets("Sensors")
    Dim nm As String, col As Long, p1 As Long, cp As Long, k As Double
    nm = wsC.Range("B4").Value: p1 = wsC.Range("B5").Value: cp = wsC.Range("B6").Value: k = wsC.Range("B7").Value
    col = SensorCol(nm)
    If col = 0 Then MsgBox "Sensor " & nm & " not found.": Exit Sub
    Dim last As Long: last = wsS.Cells(Rows.Count, 1).End(xlUp).Row
    Dim rc As Long: rc = cp + 1  ' sheet row of change-point run (run_id n sits in row n+1)
    Dim cl1 As Double, l1 As Double, u1 As Double, m1 As Double
    Dim cl2 As Double, l2 As Double, u2 As Double, m2 As Double
    Limits col, FIRST_ROW, p1 + 1, k, cl1, l1, u1, m1
    Dim endRow As Long: endRow = rc + 199: If endRow > last Then endRow = last
    Limits col, rc, endRow, k, cl2, l2, u2, m2

    Dim r As Long, v As Variant, a1 As Long, a2 As Long, n As Long
    For r = rc To last
        v = wsS.Cells(r, col).Value
        If IsNumeric(v) And Not IsEmpty(v) Then
            n = n + 1
            If v > u1 Or v < l1 Then a1 = a1 + 1
            If v > u2 Or v < l2 Then a2 = a2 + 1
        End If
    Next r
    Dim days As Double: days = wsS.Cells(last, 2).Value - wsS.Cells(rc, 2).Value
    If days < 1 Then days = 1
    wsC.Range("B18").Value = n: wsC.Range("B19").Value = a1: wsC.Range("B20").Value = a2
    wsC.Range("B21").Value = Round(days, 1)
    wsC.Range("B22").Value = Round(a1 / days, 2): wsC.Range("B23").Value = Round(a2 / days, 2)
    MsgBox nm & " after run " & cp & ": " & Format(a1 / days, "0.00") & " alarms/day with frozen limits, " & _
           Format(a2 / days, "0.00") & " with re-baselined limits.", vbInformation, "SECOM SPC"
End Sub
