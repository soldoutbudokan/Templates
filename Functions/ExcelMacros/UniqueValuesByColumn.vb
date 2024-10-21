' The following macro iterates through each column selected in some range and returns the unique values in that range by column. 
' The range must be contiguous, but does not need to include the entire column. By default it will include headers. If you wish to not include headers
' don't select that as part of the range

Sub UniqueValuesByColumn()
    Dim sourceRange As Range
    Dim destinationSheet As Worksheet
    Dim uniqueValues As Object
    Dim cell As Range
    Dim col As Range
    Dim i As Long, j As Long
    Dim sheetName As String
    
    ' Get the active selection
    Set sourceRange = Selection
    
    ' Check if a range was selected
    If sourceRange Is Nothing Then
        MsgBox "No range selected. Please select a range and run the macro again.", vbExclamation
        Exit Sub
    End If
    
    ' Check if the selection is a contiguous range
    If Not IsContiguousRange(sourceRange) Then
        MsgBox "Please select a contiguous range. The current selection contains gaps or is irregular.", vbExclamation
        Exit Sub
    End If
    
    ' Create a new sheet with a unique name
    i = 1
    Do
        sheetName = "Unique Values " & i
        Set destinationSheet = Nothing
        On Error Resume Next
        Set destinationSheet = ThisWorkbook.Worksheets(sheetName)
        On Error GoTo 0
        If destinationSheet Is Nothing Then
            Set destinationSheet = ThisWorkbook.Worksheets.Add
            destinationSheet.Name = sheetName
            Exit Do
        End If
        i = i + 1
    Loop
    
    ' Process each column in the selected range
    For j = 1 To sourceRange.Columns.Count
        Set col = sourceRange.Columns(j)
        
        ' Create a dictionary to store unique values
        Set uniqueValues = CreateObject("Scripting.Dictionary")
        
        ' Populate the dictionary with unique values
        For Each cell In col.Cells
            If Not IsEmpty(cell) Then
                If Not uniqueValues.Exists(cell.Value) Then
                    uniqueValues.Add cell.Value, 1
                End If
            End If
        Next cell
        
        ' Output unique values to the new sheet
        i = 1
        For Each Key In uniqueValues.Keys
            destinationSheet.Cells(i, j).Value = Key
            i = i + 1
        Next Key
    Next j
    
    ' Autofit the columns
    destinationSheet.Columns.AutoFit
    
    MsgBox "Unique values have been extracted to a new sheet named '" & sheetName & "'.", vbInformation
End Sub

Function IsContiguousRange(rng As Range) As Boolean
    IsContiguousRange = (rng.Areas.Count = 1)
End Function